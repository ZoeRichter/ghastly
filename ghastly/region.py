import numpy as np


class Region:
    '''
    Parent class for Region objects.
    '''

    def __init__(self, x_c, y_c, zmax, zmin, reg_id,
                 inlet, outlet, start=False, end=False):
        '''
        Initializes a single instance of a Region object.  All dimensions are
        in meters. This does not define a specific geometry, and should not be
        used directly.

        Parameters
        ----------
        x_c : float
            Coordinate of the cylinder's center on the x-axis [m].
        y_c : float
            Coordinate of the cylinder's center on the y-axis [m].
        zmax : float
            Z-coordinate of the cylinder's top [m].
        zmin : float
            Z-coordinate of the cylinder's bottom [m].
        reg_id : str
            Unique ID for this region.
        inlet : list
            List of Region IDs that feed into this region.  A region may have
            more than one inlet.
        outlet : str
            Region ID for the region that this one feeds into.  A region may only
            have one outlet.
        start : bool
            True if this region is part of the "start" of a pebble's
            path in the core - for example, if it is in the top of an HTGR -
            False otherwise.
        end : bool
            True if this region is at the "end" of the pebble pathway through
            the core - i.e., if it is the discharge region - False otherwise
        '''
        self.x_c = x_c
        self.y_c = y_c
        self.zmax = zmax
        self.zmin = zmin
        self.h = abs(zmin) + abs(zmax)
        self.reg_id = reg_id
        self.inlet = inlet
        self.outlet = outlet
        self.start = start
        self.end = end


class CylReg(Region):
    '''
    Class for a cylindrical region aligned with the z-axis.
    '''

    def __init__(self, r, *args, **kwargs):
        '''
        Initializes a single instance of a CylReg object.  All dimensions
        should be in meters.

        Parameters
        ----------
        r : float
            Radius of the cylinder [m].
        '''
        super().__init__(*args, **kwargs)
        self.r = r
        self.volume = np.pi * (r**2) * self.h


class AnnularReg(Region):
    '''
    Class for a sector of an annular region, to be used with CylCore objects.
    '''

    def __init__(self, r_outer, r_inner, theta_min=0, theta_max=2 * np.pi,
                 *args, **kwargs):
        '''
        Initializes a single instance of an AnnularReg object.  All distances
        should be in meters, all angles in radians.

        Parameters
        ----------
        r_outer : float
            Outer radius of the annulus [m].
        r_inner : float
            Inner radius of the annulus [m].
        theta_min : float
            Smaller angle that defines the sector of the annular region.
            Default is 0 radians.
        theta_max : float
            Larger angle that defines the sector of the annular region.
            Default is 2pi radians.
        '''
        super().__init__(*args, **kwargs)
        self.r_outer = r_outer
        self.r_inner = r_inner
        self.theta_min = theta_min
        self.theta_max = theta_max
        self.volume = ((np.pi * (r_outer**2 - r_inner**2) * self.h)
                       * ((theta_max - theta_min) / (2 * np.pi)))


class ConeReg(Region):
    '''
    Class for a region in the shape of a truncated right cone.
    '''

    def __init__(self, r_major, r_minor, *args, **kwargs):
        '''
        Initializes a ConeReg object.  All dimensions should be in meters.

        Parameters
        ----------
        r_major : float
            Radius at the top of the cone [m].
        r_minor : float
            Radius at the bottom of the cone [m].
        '''
        super().__init__(*args, **kwargs)
        self.r_major = r_major
        self.r_minor = r_minor
        self.volume = ((1 / 3) * np.pi * self.h
                       * (r_major**2 + r_minor**2 + r_major * r_minor))


class AnnConeReg(Region):
    '''
    Class for a region in the shape of a truncated annular cone.
    '''

    def __init__(self, r_out_up, r_in_up, r_out_low, r_in_low,
                 *args, **kwargs):
        '''
        Initializes an AnnCoreReg object.  All dimensions should be in meters.

        Parameters
        ----------
        r_out_up : float
            Outer radius at the top of the region [m].
        r_in_up : float
            Inner radius at the top of the region [m].
        r_out_low : float
            Outer radius at the bottom of the region [m].
        r_in_low : float
            Inner radius at the bottom of the region [m].
        '''
        super().__init__(*args, **kwargs)
        self.r_out_major = r_out_major
        self.r_in_major = r_in_major
        self.r_out_minor = r_out_minor
        self.r_in_minor = r_in_minor
        self.volume = ((1 / 3) * np.pi * self.h * (
            (r_out_major**2 + r_out_minor**2 + r_out_major * r_out_minor)
            - (r_in_major**2 + r_in_minor**2 + r_in_major * r_in_minor)))
