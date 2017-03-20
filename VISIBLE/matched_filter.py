import numpy as np
from vis_sample import vis_sample
import matplotlib.pylab as pl
import sys
from vis_sample.file_handling import *
from scipy import ndimage
import time

def matched_filter(filterfile=None, datafile=None, mu_RA=0., mu_DEC=0., src_distance=None, interpolate=True, outfile=None, mode='channel', restfreq=None, plot=False, verbose=False):
    """Applies an approximated matched filter to interferometric data to boost SNR

    matched_filter allows you to apply an approximated matched filter to weak line data and extract a stronger signal. 

    The filter can be created from a FITS image or RADMC3D output image, and the weak line data can be a CASA MS or uvfits file.

    The filter response can be output either to a .npy file or returned back to the user (for scripting)


    Parameters
    __________
    filterfile : the input filter image, needs to be in a valid FITS format with units of DEG for the RA and DEC, a RADMC3D image.out file (ascii format), or a SkyImage object from vis_sample. Must have an accurate reference frequency

    datafile - uvfits file or CASA measurement set containing the weak line. This should be as broad as possible (for a baseline), and NOT just a small window around the line

    mu_RA - (optional, default = 0) right ascension offset from phase center in arcseconds (i.e. filter visibilities are sampled as if the image is centered at (mu_RA, mu_DEC)

    mu_DEC - (optional, default = 0) declination offset from phase center in arcseconds (i.e. filter visibilities are sampled as if the image is centered at (mu_RA, mu_DEC)
 
    src_distance - distance to source in parsecs - only required for RADMC3D input images

    interpolate - (optional, default = True) whether the filter is interpolated to match the the local velocity spacing of the data. Should remain true unless you have a good reason otherwise.

    outfile - (optional) name of output file for filter response, needs to have a .npy extension

    mode - (optional, default = 'channel') output format of the x axis of the impulse response spectrum. Options are 'channel', 'frequency', and 'velocity'. 'velocity' requires a rest frequency input

    restfreq - (optional) rest frequency for 'velocity' output mode, input as a float in MHz

    verbose - (boolean) flag to print all progress output and timing


    Usage:
    __________
    >> from VISIBLE import matched_filter                                                                           # import the matched_filter command  

    >> matched_filter(filterfile="my_filter.fits", datafile="observations.ms", outfile="spectrum.npy")              # filter observations.ms using the filter image from my_filter.fits and output spectrum to spectrum.npy

    >> spectrum = matched_filter(filterfile="my_filter.fits", datafile="observations.ms")                           # filter observations.ms using the filter image from my_filter.fits, result stored in variable spectrum

    In the second usage, interp_vis is a numpy array, i.e. [frequencies, spectrum]


    >> spectrum = matched_filter(filterfile="my_filter.fits", datafile="observations.ms.cvel", mode="frequency")         # same as above, output with x axis in units of frequency. Input ms should be run through cvel prior to filtering

    >> spectrum = matched_filter(filterfile="my_filter.fits", datafile="observations.ms.cvel", mode="velocity", restfreq="235000")         # same as above, output with x axis in units of lsrk velocity. Requires input of a rest frequency, and input ms should be run through cvel prior to filtering
    """

    # Error/warning cases #
    if not filterfile:
        print "ERROR: Please supply an input filter image"
        return 

    if not datafile:
        print "ERROR: Please supply an input MS or uvfits file to filter"
        return 

    if mode=='velocity':
        if not restfreq:
            print "ERROR: Please supply a rest frequency in order to calculate output velocities"
            return 

    if mode=='velocity':
        print "WARNING: ALMA does not Doppler track, make sure that the datafile has been run through cvel or velocities will not be correct"

    if mode=='frequency':
        print "WARNING: ALMA does not Doppler track, make sure that the datafile has been run through cvel or frequencies will not be correct"


    #################################
    #   data visibility retrieval   #
    #################################

    # read visibilities in from the data file
    if verbose:
        print "Reading data file: "+datafile
        t0 = time.time()

    try:
        data = import_data_uvfits(datafile)
    except IOError:
        try:
            data = import_data_ms(datafile)
        except RuntimeError:
            print "Not a valid data file. Please check that the file is a uvfits file or measurement set"
            sys.exit(1)

    nvis = data.VV.shape[0]

    if verbose: 
        t1 = time.time()
        print "Read data file: "+datafile
        print "Data read time = " + str(t1-t0)
        

    print data.freqs[0]
    print data.freqs[-1]


    #############################
    #   Read the filter image   #
    #############################

    # now that we have the data, let's import the filter file
    if verbose:
        print "Reading filter file: "+filterfile
        t0 = time.time()

    if isinstance(filterfile, SkyImage):
        filter_img = filterfile
    elif "image.out" in filterfile:
        if src_distance is None:
             print "A source distance in pc needs to be provided in order to process a RADMC3D image file"
             return 
        else: filter_img = import_model_radmc(src_distance, filterfile)
    elif "fits" in filterfile:
        filter_img = import_model_fits(filterfile)
    else:
        print "Not a valid filter image option. Please provide a FITS file, a RADMC3D image file, or a SkyImage object)."
        return 

    # the number of filter channels needs to be smaller than the data channels
    if (len(filter_img.freqs) >= len(data.freqs)):
        print "Number of channels in filter exceeds number of data channels. Filtering cannot continue."
        return

    elif (len(filter_img.freqs) >= len(data.freqs)*0.5):
        print "WARNING: Number of channels in data file seems small compared to width of filter. Make sure there is adequate baseline in the data file."

    if verbose: 
        t1 = time.time()
        print "Read filter image: "+datafile
        print "Filter read time = " + str(t1-t0)



    ##############################
    #   Interpolate the filter   #
    ##############################

    # if interpolation enabled, then make filter match data resolution (in velocity space)
    if interpolate:
        if verbose:
            print "Interpolating filter"
            t0 = time.time()

        # determine the reference frequencies and freq spacings
        filter_rfreq = np.mean(filter_img.freqs)
        filter_delfreq = filter_img.freqs[1] - filter_img.freqs[0]

        data_rfreq = np.mean(data.freqs)
        data_delfreq = data.freqs[1] - data.freqs[0]

        if data_delfreq < 0:
            if filter_delfreq > 0:
                filter_img.data = filter_img.data[:,:,::-1]
                filter_delfreq = -filter_delfreq
        else:
            if filter_delfreq < 0:
                filter_img.data = filter_img.data[:,:,::-1]
                filter_delfreq = -filter_delfreq

        filter_vwidth = filter_delfreq/filter_rfreq*c_kms
        data_vwidth = data_delfreq/data_rfreq*c_kms

        nchan_filter = len(filter_img.freqs)
        nchan_data = len(data.freqs)

        chan_grid = np.arange(nchan_filter)
        interp_chans = (np.arange(nchan_data)*data_vwidth/filter_vwidth)[(np.arange(nchan_data)*data_vwidth/filter_vwidth) <= np.max(chan_grid)]

        interp_grid_x, interp_grid_y, interp_grid_chan = np.meshgrid(np.arange(filter_img.data.shape[0]), np.arange(filter_img.data.shape[1]), interp_chans)

        interp_grid_x = np.ravel(interp_grid_x)
        interp_grid_y = np.ravel(interp_grid_y)
        interp_grid_chan = np.ravel(interp_grid_chan)

        interp_data = ndimage.map_coordinates(filter_img.data, [interp_grid_y, interp_grid_x, interp_grid_chan], order=1)
        interp_data = interp_data.reshape((filter_img.data.shape[0], filter_img.data.shape[1], interp_chans.shape[0]))

        filter_img.data = interp_data
        filter_img.freqs = ndimage.map_coordinates(filter_img.freqs, [interp_chans], order=1)

        if verbose: 
            t1 = time.time()
            print "Filter interpolated"
            print "Filter interpolation time = " + str(t1-t0)

        print filter_img.data.shape



    #########################################
    #   Calculate the filter visibilities   #
    #########################################

    if verbose:
        print "Generating kernel"
        t0 = time.time()

    nchan_kernel = len(filter_img.freqs)

    kernel = np.empty(nchan_kernel*nvis, dtype='complex128').reshape(nvis, nchan_kernel)
    kernel[:,:] = vis_sample(imagefile=filter_img, uu=data.uu, vv=data.vv, mu_RA=mu_RA, mu_DEC=mu_DEC)

    if verbose: 
        t1 = time.time()
        print "Kernel generated"
        print "Kernel generation time = " + str(t1-t0)

    print kernel.shape



    ###############################
    #   Do the actual filtering   #
    ###############################

    if verbose:
        print "Starting kernel convolution"
        t0 = time.time()

    xc = np.zeros((data.VV.shape[1] - nchan_kernel + 1), dtype='complex128')  
    for i in np.arange(nvis):
        xc += np.correlate(data.VV[i], kernel[i])*data.wgts[i]/np.sqrt(np.dot(kernel[i],kernel[i].conj())*data.wgts[i])

    if verbose: 
        t1 = time.time()
        print "Data filtered"
        print "Kernel convolution time = " + str(t1-t0)


    '''

    ##########################
    #   Normalize response   #
    ##########################
    
    dummy_xc_real = np.real(xc)
    noise = np.copy(xc)
    dummy_xc_mean = np.mean(dummy_xc_real)
    dummy_xc_std = np.std(dummy_xc_real)

    for chan in np.where(dummy_xc_real < (dummy_xc_mean - dummy_xc_std*4))[0]:
        print chan
        noise[chan-10:chan+10] = np.nan

    for chan in np.where(dummy_xc_real > (dummy_xc_mean + dummy_xc_std*4))[0]:
        noise[chan-10:chan+10] = np.nan

    noise_mean = np.nanmean(noise)
    noise_std = np.nanstd(np.real(noise))

    xc = xc - noise_mean
    xc = xc/noise_std

    print "max signal = " + str(np.max(np.real(xc))) + " sigma"

    '''


    ###############################
    #   Calculate output x-axis   #
    ###############################

    if mode=='channel':
        response_chans = np.arange(xc.shape[0]) + nchan_kernel/2 + 0.5
        x_axis = response_chans

    elif mode=='frequency':
        response_freqs = (np.squeeze(data.freqs[nchan_kernel/2:-nchan_kernel/2+1]) + data_delfreq/2.0)/1.e6
        x_axis = response_freqs

    else:
        response_freqs = (np.squeeze(data.freqs[nchan_kernel/2:-nchan_kernel/2+1]) + data_delfreq/2.0)/1.e6
        response_vels = (restfreq - response_freqs)/restfreq*c_kms
        x_axis = response_vels




    ############################
    #   Plot filter response   #
    ############################

    if plot==True:

        fig = pl.figure(figsize=(5,2.7), dpi=300)
        ax = pl.axes([0.12,0.13,0.85,0.84])

        if mode=='channel':
            pl.plot(response_chans, np.real(xc))

        elif mode=='frequency':
            pl.plot(response_freqs, np.real(xc))

        else:
            pl.plot(response_vels, np.real(xc))

        ax.minorticks_on()

        pl.setp(ax.get_xticklabels(), size='9')
        pl.setp(ax.get_yticklabels(), size='9')

        pl.ylabel(r'Impulse response', size=9)
        ax.yaxis.set_label_coords(-0.09, 0.5)


        if mode=='channel':
            pl.xlabel(r'MS channel', size=9)

        elif mode=='frequency':
            pl.xlabel(r'Frequency [MHz]', size=9)

        else:
            pl.xlabel(r'Velocity [km s$^{-1}$]', size=9)

        ax.xaxis.set_label_coords(0.5, -0.09)

        pl.show()



    ########################
    #   Now return stuff   #
    ########################

    # simplest case is just writing to a file:
    if outfile:
        # save it
        np.save(outfile, np.vstack((x_axis, xc)))    
    

    # otherwise we're going to return the raw output of the filtering
    else:
        return np.vstack((x_axis, xc))

    return
