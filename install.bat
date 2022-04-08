rd /s /q "./env"
md "./env"
call conda env create --file "./requirements.yml" --prefix "./env"
