from dataclasses import dataclass, field
from typing import BinaryIO, Dict, Optional, Union

import blobfile as bf
import numpy as np

from .ply_util import write_ply


@dataclass
class TriMesh:
    """
    A 3D triangle mesh with optional data at the vertices and faces.
    """

    # [N x 3] array of vertex coordinates.
    verts: np.ndarray

    # [M x 3] array of triangles, pointing to indices in verts.
    faces: np.ndarray

    # [P x 3] array of normal vectors per face.
    normals: Optional[np.ndarray] = None

    # Extra data per vertex and face.
    # field(default_factory=dict):
    # This part of the line sets the default value for vertex_channels using the field function from the dataclasses module.
    # In this case, the default value is an empty dictionary ({}) created by calling the dict function.
    vertex_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)
    face_channels: Optional[Dict[str, np.ndarray]] = field(default_factory=dict)

    @classmethod
    def load(cls, f: Union[str, BinaryIO]) -> "TriMesh":
        """
        Load the mesh from a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "rb") as reader:
                return cls.load(reader)
        else:
            obj = np.load(f)
            keys = list(obj.keys())
            verts = obj["verts"]
            faces = obj["faces"]
            normals = obj["normals"] if "normals" in keys else None
            vertex_channels = {}
            face_channels = {}
            for key in keys:
                if key.startswith("v_"):
                    vertex_channels[key[2:]] = obj[key]
                elif key.startswith("f_"):
                    face_channels[key[2:]] = obj[key]
            return cls(
                verts=verts,
                faces=faces,
                normals=normals,
                vertex_channels=vertex_channels,
                face_channels=face_channels,
            )

    def save(self, f: Union[str, BinaryIO]):
        """
        Save the mesh to a .npz file.
        """
        if isinstance(f, str):
            with bf.BlobFile(f, "wb") as writer:
                self.save(writer)
        else:
            obj_dict = dict(verts=self.verts, faces=self.faces)
            if self.normals is not None:
                obj_dict["normals"] = self.normals
            # By convention, the author has chosen to use the prefixes "v_" and "f_" to identify different types of data in the file.
            # This helps in distinguishing between vertex-related data and face-related data,
            # making it easier to process and assign them to the appropriate variables (vertex_channels and face_channels, respectively).
            for k, v in self.vertex_channels.items():
                obj_dict[f"v_{k}"] = v
            for k, v in self.face_channels.items():
                obj_dict[f"f_{k}"] = v
            np.savez(f, **obj_dict)

    def has_vertex_colors(self) -> bool:
        return self.vertex_channels is not None and all(x in self.vertex_channels for x in "RGB")

    def write_ply(self, raw_f: BinaryIO):
        write_ply(
            raw_f,
            coords=self.verts,
            rgb=(
                np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
                if self.has_vertex_colors()
                else None
            ),
            faces=self.faces,
        )

    def write_obj(self, raw_f: BinaryIO) -> None:
        """
        Writes the mesh data in OBJ format to a binary file.

        Args:
            raw_f (BinaryIO): Binary file object to write the OBJ data to.
        """

        if self.has_vertex_colors():
            # If the mesh has vertex colors, create vertex color data
            vertex_colors = np.stack([self.vertex_channels[x] for x in "RGB"], axis=1)
            vertices = [
                "{} {} {} {} {} {}".format(*coord, *color)
                for coord, color in zip(self.verts.tolist(), vertex_colors.tolist())
            ]
        else:
            # If the mesh doesn't have vertex colors, only create vertex position data
            vertices = ["{} {} {}".format(*coord) for coord in self.verts.tolist()]

        # Convert the faces into OBJ format
        """
        
The + 1 operation in the line "f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1)) 
is used to adjust the indices of the vertices in the OBJ format.

In the OBJ file format, vertex indices start from 1, rather than 0 as in many programming languages. 
Therefore, when writing the face data to the OBJ file, the indices of the vertices need to be incremented by 1 to match the OBJ format's indexing convention.

The tri variable in the line represents a triplet of vertex indices that form a face. 
By adding 1 to each index (tri[0] + 1, tri[1] + 1, tri[2] + 1), the vertex indices are adjusted accordingly before being formatted into the OBJ face format string.

For example, if a face has vertex indices [0, 1, 2], the line "f {} {} {}".format(str(tri[0] + 1), 
str(tri[1] + 1), str(tri[2] + 1)) will format it as "f 1 2 3", which conforms to the OBJ format's indexing scheme.

This adjustment ensures that when the OBJ file is read by other software or parsers that follow the OBJ specification, 
the vertices will be correctly referenced by their indices, starting from 1.
        """
        faces = [
            "f {} {} {}".format(str(tri[0] + 1), str(tri[1] + 1), str(tri[2] + 1))
            for tri in self.faces.tolist()
        ]

        # Combine the vertex and face data
        combined_data = ["v " + vertex for vertex in vertices] + faces

        # Write the combined data to the file
        raw_f.writelines("\n".join(combined_data))
