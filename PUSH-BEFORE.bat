IF exist ..\tmp (echo tmp already exists) ELSE (cd .. && mkdir tmp && echo tmp created)
move checkpoints ..\tmp
move out ..\tmp
move sample ..\tmp
move *.m4v ..\tmp
move *.avi ..\tmp
del /Q /S *.bak
del /Q /S *.pyc
pause