# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import os

compiler_file = os.path.realpath(f"{os.getcwd()}/../compiler")
if not os.path.exists(compiler_file):
    raise RuntimeError(f"{compiler_file } file not found")
with open(compiler_file) as fp:
    compile_cmd = fp.read().replace("\n", " ").strip()
