conda clean -a -y
md "../env"
call conda remove -p "../env/mimosa" --force-remove -y --all
md "../env/mimosa"
call conda env create --file "./requirements.yml" --prefix "../env/mimosa"
