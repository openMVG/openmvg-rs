// SPDX-FileCopyrightText: 2026 Pierre Moulon and OpenMVG Rust contributors
// SPDX-License-Identifier: MIT OR Apache-2.0

/// Implement by-value `with_*` setters and same-name getters for private config fields.
///
/// A `copy` field gets a `const` getter returning the value; a `ref` field gets a getter
/// returning a reference. A field's doc comment documents both methods.
macro_rules! config_accessors {
    ($ty:ty { $($(#[$doc:meta])* $kind:ident $field:ident, $with:ident: $field_ty:ty;)* }) => {
        impl $ty {
            $(config_accessors!(@$kind $(#[$doc])* $field, $with: $field_ty);)*
        }
    };
    (@copy $(#[$doc:meta])* $field:ident, $with:ident: $field_ty:ty) => {
        $(#[$doc])*
        #[must_use]
        pub const fn $with(mut self, $field: $field_ty) -> Self {
            self.$field = $field;
            self
        }

        $(#[$doc])*
        #[must_use]
        pub const fn $field(&self) -> $field_ty {
            self.$field
        }
    };
    (@ref $(#[$doc:meta])* $field:ident, $with:ident: $field_ty:ty) => {
        $(#[$doc])*
        #[must_use]
        pub fn $with(mut self, $field: $field_ty) -> Self {
            self.$field = $field;
            self
        }

        $(#[$doc])*
        #[must_use]
        pub const fn $field(&self) -> &$field_ty {
            &self.$field
        }
    };
}

#[cfg(test)]
mod tests {
    #[cfg(not(feature = "std"))]
    use alloc::{borrow::ToOwned, string::String};

    #[derive(Default)]
    struct Probe {
        count: usize,
        name: String,
    }

    config_accessors!(Probe {
        /// A copied field.
        copy count, with_count: usize;
        /// A borrowed field.
        ref name, with_name: String;
    });

    #[test]
    fn accessors_set_and_read_each_field() {
        let probe = Probe::default().with_count(3).with_name("probe".to_owned());
        assert_eq!(probe.count(), 3);
        assert_eq!(probe.name(), "probe");
    }
}
