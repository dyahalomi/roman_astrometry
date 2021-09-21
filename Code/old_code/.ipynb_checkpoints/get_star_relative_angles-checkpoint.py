def get_star_relative_angles(self, t, parallax=None, light_delay=False):
    """The stars' relative position to the star in the sky plane, in
    separation, position angle coordinates.
    .. note:: This treats each planet independently and does not take the
        other planets into account when computing the position of the
        star. This is fine as long as the planet masses are small.
    Args:
        t: The times where the position should be evaluated.
        light_delay: account for the light travel time delay? Default is
            False.
    Returns:
        The separation (arcseconds) and position angle (radians,
        measured east of north) of the planet relative to the star.
    """
    X, Y, Z = self._get_star_position(
        -self.a, t, parallax, light_delay=light_delay
    )

    # calculate rho and theta
    rho = tt.squeeze(tt.sqrt(X ** 2 + Y ** 2))  # arcsec
    theta = tt.squeeze(tt.arctan2(Y, X))  # radians between [-pi, pi]

    return (rho, theta)



def test_relative_angles():
    #test seperation and position angle with Earth and Sun
    p_earth=365.256
    t = np.linspace(0, 1000, 1000)
    m_earth = 1.*3.00273e-6 #units m_sun
    orbit_earth = xo.orbits.KeplerianOrbit(
        m_star=1.,
        r_star=1.,
        t0=0.5,
        period=p_earth,
        ecc=0.0167,
        omega=np.radians(102.9),
        Omega=np.radians(0.0),
        incl=np.radians(45.0),
        m_planet=m_earth,
    )


    rho_star_earth, theta_star_earth = theano.function([], orbit_earth._get_star_relative_angles(t, parallax=0.1))()
    rho_earth, theta_earth = theano.function([], orbit_earth._get_relative_angles(t, parallax=0.1))()

    rho_star_earth_diff = np.max(rho_star_earth) - np.min(rho_star_earth)
    rho_earth_diff = np.max(rho_earth)- np.min(rho_earth)


    #make sure amplitude of separation is correct for star and planet motion
    assert np.isclose(rho_earth_diff, 3.0813126e-02)
    assert np.isclose(rho_star_earth_diff, 9.2523221e-08)


    #make sure planet and star position angle closely mirrors each other
    assert np.allclose(theta_earth[:int(p_earth/2)], theta_star_earth[int(p_earth/2):int(p_earth)-1], atol=0.2)



    #test seperation with Jupiter and Sun
    p_jup=4327.631
    t = np.linspace(0, 10000, 10000)
    m_jup = 317.83*3.00273e-6 #units m_sun
    orbit_jup = KeplerianOrbit(
        m_star=1.,
        r_star=1.,
        t0=2000,
        period=p_jup,
        ecc=0.0484,
        omega=np.radians(274.3) - 2*np.pi,
        Omega=np.radians(100.4),
        incl=np.radians(45.0),
        m_planet=m_jup,
    )


    rho_star_jup, theta_star_jup = theano.function([], orbit_jup._get_star_relative_angles(t, parallax=0.1))()
    rho_jup, theta_jup = theano.function([], orbit_jup._get_relative_angles(t, parallax=0.1))()

    rho_star_earth_diff = np.max(rho_star_jup) - np.min(rho_star_jup)
    rho_earth_diff = np.max(rho_jup)- np.min(rho_jup)

    
    #make sure amplitude of separation is correct for star and planet motion
    assert np.isclose(rho_jup_diff, 1.7190731e-01)
    assert np.isclose(rho_star_jup_diff, 1.6390463e-04)


    #make sure planet and star position angle closely mirrors each other
    assert np.allclose(theta_jup[:int(p_jup/2)], theta_star_jup[int(p_jup/2):int(p_jup)-1], atol=0.2)



    


