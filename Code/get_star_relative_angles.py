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


