==================================================================
 PhasePix Portable - Windows Zero-Install Distribution
==================================================================

PhasePix is a quantitative SEM microstructure analyzer. This folder
contains everything it needs: Python, Git, all scientific libraries
and the base AI model weights. Nothing is installed on your PC.

------------------------------------------------------------------
 HOW TO START
------------------------------------------------------------------
1. Double-click  PhasePix_Start.bat
2. Your browser opens at  http://localhost:8501
3. Keep the black console window open while you work.
   Closing it stops the PhasePix server.

Optional: run Create_Desktop_Shortcut.bat once to place a PhasePix
icon on your Desktop with the custom Icon.ico artwork.

------------------------------------------------------------------
 YOUR MICROGRAPHS
------------------------------------------------------------------
The images folders always start empty - no laboratory data is
shipped inside this distribution.

Double-click  Open_Images_Folder.bat  to open them directly:
app\images\sem\    for SEM images
app\images\light\  for light microscopy images

Copies of your files stay in these folders and survive every
automatic code update.

------------------------------------------------------------------
 AUTOMATIC UPDATES
------------------------------------------------------------------
Every launch pulls the latest code from GitHub:

  * Your edits in app\parameter_presets.json and
    app\light_calibration_presets.json are backed up and restored,
    so laboratory recipes are never lost.
  * Without an internet connection PhasePix simply starts the
    version that is already in this folder.

------------------------------------------------------------------
 FOLDER LAYOUT
------------------------------------------------------------------
python\                 Portable CPython 3.11 + site-packages
git\                    Portable MinGit 2.47.0
app\                    PhasePix source code - a git clone
app\images\             Put your micrographs here
                        images\sem\   for SEM images
                        images\light\ for light microscopy images
app\models_cache\       AI weights - Cellpose and Segment Anything
PhasePix_Start.bat      Main launcher
Create_Desktop_Shortcut.bat   Desktop icon helper
Open_Images_Folder.bat  Opens app\images in Windows Explorer

------------------------------------------------------------------
 TROUBLESHOOTING
------------------------------------------------------------------
* Antivirus warning: python\python.exe and git\cmd\git.exe are
  portable, unsigned binaries. Allow them or whitelist this folder.
* Slow first start: the AI model loads into memory the first time
  an image is analyzed. Later runs are much faster.
* No GPU: the CUDA build of PyTorch also runs on plain CPUs, just
  slower. No extra download is needed.
* Port already in use: close the other PhasePix window, or edit the
  --server.port value inside PhasePix_Start.bat.
* Rebuild needed: run build_portable_windows.bat on a build PC and
  copy the whole PhasePix_Portable folder again.

==================================================================
