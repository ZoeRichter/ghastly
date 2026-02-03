import numpy as np


class Core:
    '''
    Parent class for Core objects.  Because the Core class doesn't define
    a specific geometry, the Core class should not be used directly.
    '''

    def __init__(self, x_c, y_c, zmax, zmin, regions=[],
                 open_bottom="open 1", open_top="open 2"):
        '''
        Initializes a single instance of a Core object.

        Parameters
        ----------
        x_c : float
            Coordinate of the core element's center on the x_axis.
        y_c : float
            Coordinate of the core element's center on the y_axis.
        zmax : float
            Z-coordinate of the core element's top.
        zmin : float
            Z-coordinate of the core element's bottom.
        regions : list
            List containing the region_id of each element within the given
            CylCore object.
        open_bottom : str
            LAMMPS parameter describing whether the bottom surface of the 
            region is open.  "open 1", the default, means that the bottom
            is open, while entering "" will input nothing, resulting in
            a closed surface.
        '''
        self.x_c = x_c
        self.y_c = y_c
        self.zmax = zmax
        self.zmin = zmin
        self.regions = regions
        self.open_bottom = open_bottom
        self.open_top = open_top
        self.h = zmax - zmin


class CylCore(Core):
    '''
    Class for a cylindrically shaped core section, with its axis parallel
    to the z-axis.
    '''

    def __init__(self, r, *args, **kwargs):
        '''
        Initializes a single instance of a CylCore object.  All dimensions
        should be in meters.

        Parameters
        ----------
        r : float
            Radius of the cylinder [m].
        '''
        super().__init__(*args, **kwargs)
        self.r = r
        self.volume = np.pi * (r**2) * self.h


class AnnularCore(Core):
    '''
    Class for an annularly shaped core region, with its axis parallel to the
    z-axis.
    '''

    def __init__(self, r_outer, r_inner, *args, **kwargs):
        '''
        Initializes an AnnularCore object.  Note that unlike an AnnularRegion,
        this assumes it is a full annular shell, not a sector.  Dimensions are
        in meters.

        Parameters
        ----------
        r_outer : float
            Outer radius of the annulus [m].
        r_inner : float
            Inner radius of the annulus [m].
        '''
        super().__init__(*args, **kwargs)
        self.r_outer = r_outer
        self.r_inner = r_inner
        self.volume = (np.pi * (r_outer**2 - r_inner**2) * self.h)
        raise NotImplementedError("Annular core regions are not implemented.")


class ConeCore(Core):
    '''
    Class for a right truncated cone, such as those found in discharge chutes.
    The central axis is parallel to the z-axis.
    '''

    def __init__(self, r_major, r_minor, *args, **kwargs):
        '''
        Intializes a single instance of a ConeCore object.  All distances are
        in meters.

        Parameters
        ----------
        r_upper: float
            Radius of the top of the cone [m].
        r_lower : float
            Radius of the bottom of the cone [m].
        '''
        super().__init__(*args, **kwargs)
        self.r_major = r_major
        self.r_minor = r_minor
        self.volume = ((1 / 3) * np.pi * self.h
                       * (r_major**2 + r_minor**2 + r_major * r_minor))


class AnnConeCore(Core):
    '''
    Class for an annular right truncated cone, with its centerline parallel
    to the z-axis
    '''

    def __init__(self, r_out_major, r_in_major, r_out_minor, r_in_minor,
                 *args, **kwargs):
        '''
        Initializes an AnnConeCore object.  All distances should be in meters.

        Parameters
        ----------
        r_out_major : float
            The outer radius at the upper part of the annular cone [m].
        r_in_major : float
            The inner radius at the upper part of the annular cone [m].
        r_out_minor : float
            The outer radius at the lower part of the annular cone [m].
        r_in_minor : float
            The inner radius at the lower part of the annular cone [m].
        '''
        super().__init__(*args, **kwargs)
        self.r_out_major = r_out_major
        self.r_in_major = r_in_major
        self.r_out_minor = r_out_minor
        self.r_in_minor = r_in_minor
        self.volume = ((1 / 3) * np.pi * self.h * (
            (r_out_up**2 + r_out_low**2 + r_out_up * r_out_low)
            - (r_in_up**2 + r_in_low**2 + r_in_up * r_in_low)))
        raise NotImplementedError("Annular core regions are not implemented.")
