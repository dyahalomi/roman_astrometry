def get_star_relative_angles(t, parallax):
    
    # determine and print the star position at desired times
    pos = theano.function([], orbit.get_star_position(t, parallax))()


    #pos = tt.sum(pos, axis=-1)

    x,y,z = pos


    # calculate rho and theta
    rho = tt.squeeze(tt.sqrt(x ** 2 + y ** 2))  # arcsec
    theta = tt.squeeze(tt.arctan2(y, x))  # radians between [-pi, pi]
    
    rho, theta = rho.eval(), theta.eval()
    
    return rho, theta


rho, theta = get_star_relative_angles(x_astrometry, parallax)


ra = rho * np.sin(theta)
dec = rho * np.cos(theta)

#ra_single = ra.eval()
#dec_single = dec.eval()

ra_orbit_sum = tt.sum(ra, axis=-1)
dec_orbit_sum = tt.sum(dec, axis=-1)

ra_orbit_sum = ra_orbit_sum.eval()
dec_orbit_sum = dec_orbit_sum.eval()


fig, ax = plt.subplots(2, 1, figsize = (15,15))


ax[0].plot(x_astrometry, ra_orbit_sum, 'o')

ax[1].plot(x_astrometry, dec_orbit_sum, 'o')