IF exist ..\tmp (echo tmp already exists) ELSE (cd .. && mkdir tmp && echo tmp created)
move checkpoints ..\tmp
move out ..\tmp
move samples ..\tmp
move database ..\tmp
move model ..\tmp
move *.m4v ..\tmp
move *.mp4 ..\tmp
move *.avi ..\tmp
del /Q /S *.bak
del /Q /S *.pyc
pause