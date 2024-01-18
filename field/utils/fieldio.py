# Copyright (c) 2024 Jeremiah Lübke <jeremiah.luebke@rub.de>,
# Frederic Effenberger, Mike Wilbert, Horst Fichtner, Rainer Grauer
#
# Distributed under the MIT License

import os
import numpy as np
import tables as tb
import itertools
from pathlib import Path
from jinja2 import Environment, FileSystemLoader


def _get_writer_kwds(kwds):
    def pop(kwds, strict, **entry):
        name, default = entry.popitem()
        value = (
            kwds.pop(name) if strict and default is None else kwds.pop(name, default)
        )
        return (name, value)

    write_field = kwds.pop("write_field", False)
    key, filename = pop(kwds, write_field, filename=None)
    filename = Path(filename).resolve() if filename is not None else None
    writer_kwds = dict(
        (
            (key, filename),
            pop(kwds, write_field, chunkshape=None),
            pop(kwds, write_field, write_xdmf=False),
            pop(kwds, write_field, note=""),
        )
    )
    return write_field, writer_kwds


def _get_h5_path(fp, kind=None, name=""):
    for idx in itertools.count(start=0):
        cand = f"{kind}/i{idx}" if kind else f"i{idx}"
        if cand + f"/{name}0" not in fp.root:
            return "/" + cand


def _get_xdmf_filename(filename, h5path):
    directory = Path(filename.parent, filename.stem + "_xdmf")
    directory.mkdir(exist_ok=True)
    h5path = h5path[1:].replace("/", "_")
    xdmf_filename = Path(directory, f"{h5path}.xdmf")
    return xdmf_filename


class FieldWriter:
    def write_field(
        self,
        *args,
        filename,
        chunkshape,
        write_xdmf=False,
        note="",
        **kwds,
    ):
        with tb.open_file(filename, "a") as fp:
            h5path = _get_h5_path(fp, self._kind, self.name)
            print(f"writing {filename}:{h5path}/{self.name}", end="")
            for i in range(self.components):
                fp.create_carray(
                    h5path,
                    f"{self.name}{i}",
                    chunkshape=chunkshape,
                    obj=self.res[i],
                    createparents=True,
                )
            print(f". writing attributes", end="")
            attrs = fp.get_node(h5path)._v_attrs
            attrs.args = [arg for arg in args if not isinstance(arg, np.ndarray)]
            attrs.kwds = kwds
            conf = dict(
                grid_size=self.grid_size,
                dimension=self.dimension,
                components=self.components,
                L_box=self.L_box,
                precision=self.precision,
                num_threads=self.num_threads,
            )
            if hasattr(self, "cfl"):
                conf.update({"cfl": self.cfl})
            attrs.conf = conf
            attrs.version = (
                os.popen("git describe --always --tags --dirty").read().strip()
            )
            attrs.note = note
        if write_xdmf:
            self.write_xdmf(filename, h5path)
        print(".")


class FieldReader:
    @staticmethod
    def _read_array(inp: tb.Array, out: np.ndarray):
        if inp.dtype != out.dtype:
            raise TypeError(
                f"Array on disk has dtype {inp.dtype}. " f"Expected dtype {out.dtype}"
            )
        if np.prod(inp.shape) != out.size:
            raise ValueError(
                f"Array on disk has shape {inp.shape}. " f"Expected shape {out.shape}"
            )
        inp.read(out=out)

    @classmethod
    def from_h5_dataset(
        cls, filename, *, idx=0, kind=None, name="B", init_pyfftw=False, **kwds
    ):
        if cls.__name__ != "BaseField":
            raise RuntimeError(
                "`from_h5_dataset` can only be called from the base class `BaseField`"
            )
        h5path = f"/{kind}/i{idx}" if kind else f"/i{idx}"
        print(f"creating new field object from {filename}:{h5path}", end="")
        with tb.open_file(filename, "r") as fp:
            node = fp.get_node(h5path)
            field = cls(
                **node._v_attrs.conf, name=name, init_pyfftw=init_pyfftw, **kwds
            )
            field._kind = kind
            for i in range(field.components):
                array = node._f_get_child(f"{name}{i}")
                FieldReader._read_array(array, field.res[i])
        print(".")
        return field

    def load_h5_dataset(self, filename, *, idx=0, kind=None, name="B"):
        h5path = f"/{kind}/i{idx}" if kind else f"/i{idx}"
        print(f"loading {filename}:{h5path} into existing buffer", end="")
        self.name = name
        with tb.open_file(filename, "r") as fp:
            for i in range(self.components):
                array = fp.get_node(h5path + f"/{name}{i}")
                FieldReader._read_array(array, self.res[i])
        print(".")
        return self.res


class XdmfWriter:
    _xdmf_template_path = str(
        Path(__file__).resolve().parent.joinpath("xdmf_templates")
    )

    def write_xdmf(self, filename, h5path):
        xdmf_file = _get_xdmf_filename(filename, h5path)
        precision = {"float32": 4, "float64": 8}[self.ftype.name]
        path_dict = {
            f"{self.name}{i}": f"{h5path}/{self.name}{i}"
            for i in range(self.components)
        }
        var_dict = {
            3: {"nx": self.grid_size, "ny": self.grid_size, "nz": self.grid_size},
            2: {"nx": self.grid_size, "ny": self.grid_size, "nz": 1},
            1: {"nx": 1, "ny": 1, "nz": self.grid_size},
        }[self.dimension] | {"dx": self.dx, "dy": self.dx, "dz": self.dx}
        env = Environment(loader=FileSystemLoader(self._xdmf_template_path))
        template = env.get_template("3d.xdmf.template")
        xdmf_str = template.render(
            **var_dict,
            filename=filename,
            precision=precision,
            paths=path_dict,
        )
        print(f".\nwriting {xdmf_file}", end="")
        with open(xdmf_file, "w+") as xfp:
            xfp.write(xdmf_str)


class FieldIO(FieldWriter, FieldReader, XdmfWriter):
    pass
