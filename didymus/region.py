import numpy as np

class Region:
    '''
    Parent class for Region objects.
    '''
    def __init__(self, reg_id, intake, outtake, start = False, end = False):
        '''
        Initializes a single instance of a Region object.  This does not
        specify any geometry, and should not be used directly.

        Parameters
        ----------
        reg_id : str
            Unique ID for this region.
        intake : list
            list of reg_ids that feed into this region.  A region may have
            more than one intake.
        outtake : str
            reg_id for the region that this one feeds into.  A region may only
            have one outtake.
        start : bool
            True if this region is part of the "start" of a pebble's
            path in the core - for example, if it is in the top of an HTGR -
            False otherwise.
        end : bool
            True if this region is at the end of the pebble pathway through
            the core - i.e., if it is the discharge region - False otherwise

        '''
        self.reg_id = reg_id
        self.intake = intake
        self.outtake = outtake
        self.start = start
        self.end = end

class AnnularReg(Region):
    '''
    Class for a sector of an annular region, to be used with CylCore objects.
    '''
    def __init__(self, x_c, y_c, r_outer, r_inner, z_max, z_min,
                 theta_min = 0, theta_max = 2*np.pi, *args, **kwargs):
        '''
        Initializes a single instance of an AnnularReg object.  All distances
        should be in meters, all angles in radians.

        Parameters
        ----------
        x_c : float
            Coordinate of the region's center on the x-axis
        y_c : float
            Coordinate of the region's center on the y_axis
        r_outer : float
            Outer radius of the annulus
        r_inner : float
            Inner radius of the annulus
        z_max : float
            Z-coordinate of the top of the annulus
        z_min : float
            Z-coordinate of the bottom of the annulus
        theta_min : float
            Smaller angle that defines the sector of the annular region.
            Default is 0 radians.
        theta_max : float
            Larger angle that defines the sector of the annular region.
            Default is 2pi radians.
        '''
        self.x_c = x_c
        self.y_c = y_c
        self.r_outer = r_outer
        self.r_inner = r_inner
        self.z_max = z_max
        self.z_min = z_min
        self.theta_min = theta_min
        self.theta_max = theta_max
        
