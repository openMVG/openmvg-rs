# openmvg-cli

`openmvg-cli` is the command-line front door of OpenMVG-rs. It installs two programs:

- `openmvg-sfm` reconstructs a folder of images with the Incremental or Global engine of
  `openmvg-sfm` and writes the cameras and points as PLY, OMVG, and optionally COLMAP;
- `openmvg-sfm-configure`, behind the non-default `tui` feature, is a terminal app that groups
  the images of a folder into cameras and writes the project file `openmvg-sfm --project` runs
  with.

```bash
cargo install openmvg-cli                  # openmvg-sfm
cargo install openmvg-cli --features tui   # openmvg-sfm and openmvg-sfm-configure
```

From a checkout, `pixi run openmvg-sfm path/to/images` and `pixi run sfm-configure path/to/images`
run them. Both are native programs: there is no WASM build.

## openmvg-sfm

```text
image folder ──► EXIF grouping, K.txt, or --project ──► Dataset
                                                           │ Frontend (SIFT or ALP, HNSW)
                                                           ▼
                                                       ViewGraph
                                                           │ Incremental or Global
                                                           ▼
             reconstruction.ply, reconstruction.omvg ◄── Reconstruction + RunReport
```

```bash
openmvg-sfm path/to/images                       # Incremental, EXIF camera groups
openmvg-sfm path/to/images --engine global --colmap
openmvg-sfm path/to/images --project path/to/images/sfm_project.json
```

The argument is an image directory, or a dataset root containing `images/`. The command uses only
the stable API of `openmvg-sfm` and `openmvg-io`, so a library user can reproduce any run.

| Option | Effect |
|---|---|
| `--engine incremental` (default), `global` | Reconstruction engine. Global returns only a reconstruction that passes its rotation and translation consistency checks, and fails otherwise |
| `--intrinsics K.txt` | A known pinhole calibration shared by every image, which stays fixed. A `K.txt` beside the images is found without the option |
| `--camera-model` | Model declared for EXIF camera groups: `pinhole`, `pinhole-radial1` (default), `pinhole-radial2`, or `spherical`; distortion starts unknown and is estimated |
| `--share-intrinsics`, `--focal-scale` | One camera for images with no EXIF camera and the same size; a factor on the EXIF or image-size focal guess |
| `--project FILE` | The grouping a project file sets, written by `openmvg-sfm-configure`; it replaces the four options above |
| `--self-calibrate true/false` | Refine or freeze the declared camera models in bundle adjustment; by default EXIF groups are refined and a project says which |
| `--detector alp` (default), `sift`; `--matcher hnsw` (default), `brute-force` | Feature detector and descriptor matcher |
| `--density` (both detectors; `ultra` is SIFT only), `--max-dimension`, `--max-pairs-per-view`, `--no-radial-calibration` | The front-end settings of `openmvg_sfm::Frontend` |
| `--seed` | Deterministic seed; the same inputs and seed give the same reconstruction |
| `--output DIR`, `--colmap`, `--no-progress` | Output directory (default `images/sfm_output`), COLMAP text model, progress bars |

The output directory receives `reconstruction.ply`, `reconstruction.omvg` (an OMVG dataset folder
that `omvg validate` checks; the format is a 0.1 draft and may change before 1.0), `colmap/sparse/0`
with `--colmap`, `intrinsic_report.json` (how each image got its initial camera), and
`sfm_report.json` (schema version 1: the grouping, the images skipped because they could not be
read, front end, seed, the engine's `RunReport`, counts, and timings). A run that writes into the
same directory again replaces the previous outputs and prints a note that it does. The command
prints one line per refined camera group and exits with 1, after writing the reports, when the
Global engine finds no reconstruction.

## openmvg-sfm-configure

`openmvg-sfm-configure` prepares an image folder for `openmvg-sfm`. It scans the folder, groups the
images into cameras the way `openmvg-sfm` would, lets you filter the images and change the
grouping, and writes the project file that `openmvg-sfm --project` runs with.

```bash
openmvg-sfm-configure path/to/images
openmvg-sfm path/to/images --project path/to/images/sfm_project.json
```

The folder is the argument of `openmvg-sfm`: an image directory, or a dataset root containing
`images/`. The project file is `sfm_project.json` in that folder unless `--project` names another.
If the file exists it is the starting point, and images it does not name start excluded;
otherwise the scan is, and `--group` chooses how:

| `--group` | A new project starts with |
|---|---|
| `auto` (default) | One camera per EXIF camera, lens, and focal length. Images that name no camera share a camera by size and are marked `(assumed)`. |
| `off` | One camera per image. |

`--preview` chooses how the image pane draws pictures (see [Preview](#preview)). `--check`
validates an existing project without opening the interface (see
[Checking a project](#checking-a-project)). `--no-tui` prints the cameras and statistics, writes
the project when the file does not exist yet, and never opens the interface; use it when stdout
is not a terminal.

### Keys

| Key | Action |
|---|---|
| `↑` `↓` `PgUp` `PgDn` `Home` `End`, click, wheel | Move through the images |
| `/` | Filter: terms are ANDed, `!term` must be absent; they look in the file name, camera, and size |
| `space`, `*`, `c`, `Esc` | Mark the image, mark every filtered image, clear the marks; `Esc` clears the filter, then the marks |
| `g` | Move the images into an existing camera |
| `n` | Move the images into a new camera that starts with the camera of the first image |
| `x` | Leave the images out of the project |
| `m` | Next camera model: `pinhole`, `pinhole_radial1`, `pinhole_radial2`, or `spherical` for 2:1 images |
| `f` | Set the focal length of the camera under the cursor, in pixels |
| `d` | Propose dividing the camera under the cursor by EXIF camera and focal-length setting; `Enter` applies it |
| `s` | Switch self-calibration on or off for the whole run |
| `a` | Return to the automatic grouping |
| `u` | Undo |
| `w`, `q` | Write the project; quit, asking first when there are unwritten changes or the project file does not exist yet (the status line shows `[not written]`) |

`g`, `n`, and `x` act on the marked images; with none marked, on the filtered images when a filter
is set; otherwise on the image under the cursor. The status line says which. Images of different
sizes cannot share a camera, and a spherical camera needs 2:1 images. A rejected edit changes
nothing. A focal length that is not read from EXIF starts as a guess, which bundle adjustment trusts
less until you set it with `f`.

The stats panel counts the images, cameras, sizes, and EXIF cameras without running the pipeline.

### Warnings and splitting

A camera that looks wrong is flagged with a warning sign in the camera list, listed first in the
stats panel, and explained in the status line when the cursor is on it. These come from the scan
alone:

| Warning | When |
|---|---|
| mixes N cameras | The images name different EXIF cameras, or some name one and some do not |
| scanned focal lengths span A to B px | The estimated focal lengths of the images differ by more than 5%, as when a zoom lens was used at several settings: one focal length cannot describe them |
| one image with the camera, size, and focal length of #N | A lone image repeats an established camera (two or more images) within 1% of its focal length, so it probably should have joined it |

`d` offers the way out of the first two. It divides the camera by EXIF camera and, within a camera,
wherever consecutive scanned focal lengths differ by more than 5%, shows one line per new camera,
and applies it only on `Enter`; the new cameras take the median scanned focal length of their
images, and `u` undoes it. A smooth zoom range with no jump over 5% stays one camera, and a
spherical camera divides by EXIF camera only. The automatic grouping already separates zoom
settings, so the warnings mostly appear after merging cameras by hand; with one camera per image
(`--group off`) there are none.

### Images that cannot be read

An image whose header cannot be read, such as the empty file a failed copy leaves, does not stop the
scan. It is left out of every project and reported: the status line names the first one when the app
opens, the stats panel counts them (`unreadable 14, left out: ...`), and `--no-tui` and `--check`
list each file with the reason. A project names only the images it includes, so `openmvg-sfm
--project` never reads the others. A project that names an unreadable image is refused, and a folder
with fewer than two readable images says how many it could read. Plain `openmvg-sfm` without a
project skips such an image too: it prints a warning naming the file and the reason, lists it under
`skipped_images` in `sfm_report.json`, and reconstructs the others.

### Checking a project

```bash
openmvg-sfm-configure path/to/images --check
```

`--check` reads the project file and the folder, changes nothing, and needs no terminal. It prints
the cameras, statistics, and warnings, and notes the images of the folder that the project leaves
out. It exits with 1 when the file does not exist or is not a valid project, names images the folder
lacks, or puts images of different sizes in one camera, and with 0 otherwise; warnings do not fail
it. It cannot be combined with `--group` or `--no-tui`.

### Preview

The pane under the camera list shows the image under the cursor. The thumbnail a camera embedded in
a JPEG appears at once, marked `(thumbnail)` in the pane's title, and a decoded copy of at most 1024
pixels on a side replaces it. Both are turned upright by the EXIF orientation; the sizes in the list
are the stored pixel sizes, which is what `openmvg-sfm` reads. The thumbnail is drawn as the camera
made it: a Canon one is 160x120 with black bars around a 3:2 picture, so it is small and blocky, a
cue for which photo this is and not for whether it is sharp. On a development machine, reading a
thumbnail took 0.3 to 0.4 ms, and the decode and shrink about 100 ms for a 6-megapixel JPEG and
about 0.5 s for a 20-megapixel one.

Decoding and encoding run on a worker thread, so moving the cursor never waits. Only the newest
request is served, and the full decode starts only after the cursor has rested for 80 ms, so holding
an arrow shows a thumbnail for every row and never queues decodes nobody sees. The last 16 pictures
are cached, and a dialog hides the picture while it is open.

| `--preview` | The pane draws |
|---|---|
| `auto` (default) | Whatever the terminal answers it supports within 0.3 s: the Kitty graphics protocol, Sixel, or the iTerm2 inline image protocol, else half blocks |
| `kitty`, `sixel`, `iterm2` | That protocol, when the terminal query picks the wrong one |
| `halfblocks` | Unicode half blocks, which every terminal draws |
| `off` | No pane |

Inside tmux, graphics need `set -g allow-passthrough on`; `--preview halfblocks` works anywhere,
including over ssh, and starts at once, while a terminal that never answers the graphics query, such
as a Linux console or a serial line, makes `auto` start 0.3 s late. On a small machine, decoding a
20-megapixel JPEG briefly takes about 130 MB, and the app otherwise stays near 20 MB; `--preview
off` keeps it near 7 MB. The tests encode a picture with each of the four protocols and check the
escape sequence it produces, and the app has been run in a pseudo-terminal with each `--preview`
value; how a given terminal then draws the sequences is `ratatui-image`'s compatibility matrix, not
something this project tests.
