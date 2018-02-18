IF exist ..\tmp (echo tmp already exists) ELSE (cd .. && mkdir tmp && echo tmp created)
move checkpoints ..\tmp
pause