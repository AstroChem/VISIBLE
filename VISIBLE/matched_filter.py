import numpy as np
from vis_sample import vis_sample
import matplotlib.pylab as pl
import sys
from vis_sample.file_handling import *
from scipy import ndimage
from scipy import sparse
import time
import math

def matched_filter(filterfile=None, datafile=None, mu_RA=0., mu_DEC=0., src_distance=None, interpolate=True, weights='renormalize', norm_chans=None, window_func='Hanning', binfactor=2, outfile=None, mode='channel', restfreq=None, plot=False, verbose=False):
    """The matched_filter() method in VISIBLE allows you to apply an approximated matched filter to interferometric spectral line data and extract a signal. 

    The filter can be created from a FITS image or RADMC3D output image, and the weak line data can be a CASA MS or uvfits file.

    The filter response can be output either to a .npy file or returned back to the user (for scripting)


    Parameters
    __________
    filterfile : input filter image or a list of filter images, needs to be in a valid FITS format with units of DEG for the RA and DEC, a RADMC3D image.out file (ascii format), or a SkyImage object from vis_sample. Must have an accurate reference frequency

    datafile - path to uvfits file or CASA measurement set containing the weak line. This should be as broad as possible (for a baseline), and NOT just a small window around the line

    mu_RA - (optional, default = 0) right ascension offset from phase center in arcseconds (i.e. filter visibilities are sampled as if the image is centered at (mu_RA, mu_DEC)

    mu_DEC - (optional, default = 0) declination offset from phase center in arcseconds (i.e. filter visibilities are sampled as if the image is centered at (mu_RA, mu_DEC)
 
    src_distance - distance to source in parsecs - only required for RADMC3D input images

    interpolate - (optional, default = True) whether the filter is interpolated to match the the local velocity spacing of the data. Should remain true unless you have a good reason otherwise.

    weights - (optional, default = 'renormalize') options are 'renormalize', 'preserve', and 'statwt'. 'renormalize' will calculate the offset (if any) between the current weights and the scatter of the visibilities, and renormalize accordingly. If 'preserve' is selected, then the data weights are assumed to be correct as-is. 'statwt' will assume that the CASA task 'statwt' was applied to the data and no renormalization will be applied. 'renormalize' should not be used if strong lines are present in the data, and the application of statwt using channels without signal will be preferable.

    norm_chans - (optional) specify signal free channels to normalize the output spectrum. Channels should be specified as a list of start/stop channel pairs (i.e. [[0,100][130,400],[450,600]]). This option should only be used if the selected 'weights' option cannot normalize the spectrum properly. Note that the channel indices are for the 'n_chan - n_kernel + 1' sized impulse response spectrum

    window_func - (optional, default = 'Hanning') the window function used in processing the time domain data, which introduces a channel correlation. A Hanning filter is used for ALMA. Can be set to 'none' for synthetic data, other options (Welch, Hamming, etc.) will be added in the future.

    binfactor - (optional, default = 2) the degree to which data was averaged/binned after the window function was applied. The default for ALMA observations after Cycle 3 is a factor of 2 (set in the OT). Valid factors are 1, 2, 3, and 4. Factors over 4 are treated as having no channel correlation.

    outfile - (optional) name of output file for filter response, needs to have a .npy extension. If n filter images are provided then n outfiles must be specified. 

    mode - (optional, default = 'channel') output format of the x-axis of the impulse response spectrum. Options are 'channel', 'frequency', and 'velocity'.

    restfreq - (optional) rest frequency for 'velocity' output mode, input as a float in MHz. If a rest frequency is not specified then the center frequency of the data will be used.

    plot - (optional) plot the real portion of the filter response spectrum against the x-axis chosen by the 'mode' parameter. The output will still be either returned or saved to 'outfile'.

    verbose - (boolean) flag to print all progress output and timing


    Usage:
    __________
    >> from VISIBLE import matched_filter                                                                           # import the matched_filter command  

    >> matched_filter(filterfile="my_filter.fits", datafile="observations.ms", outfile="spectrum.npy")              # filter observations.ms using the filter image from my_filter.fits and output spectrum to spectrum.npy

    >> output = matched_filter(filterfile="my_filter.fits", datafile="observations.ms")                           # filter observations.ms using the filter image from my_filter.fits, result stored in variable 'output', where output looks likes [channels, xc_spectrum].

    >> spectrum = matched_filter(filterfile="my_filter.fits", datafile="observations.ms.cvel", mode="frequency")         # same as above, output with x axis in units of frequency. Input ms should be run through cvel prior to filtering

    >> spectrum = matched_filter(filterfile="my_filter.fits", datafile="observations.ms.cvel", mode="velocity")         # same as above, output with x axis in units of lsrk velocity. Input ms should be run through cvel prior to filtering
    """

    # Error/warning cases #
    if not filterfile:
        print "ERROR: Please supply an input filter image or list of filter images"
        return 

    if not datafile:
        print "ERROR: Please supply an input MS or uvfits file to filter"
        return 

    if mode=='velocity':
        print "WARNING: ALMA does not Doppler track, make sure that the datafile has been run through cvel or velocities will not be correct"

    if mode=='frequency':
        print "WARNING: ALMA does not Doppler track, make sure that the datafile has been run through cvel or frequencies will not be correct"

    if (window_func != "Hanning") and (window_func != "none"):
        print 'ERROR: Please specify a valid window function. Options are "Hanning" or "none".'
        return

    if not (type(binfactor) is int):
        print 'ERROR: Please specify a valid binning factor. Value should be a positive integer and values greater than 4 will result in data being treated as having no channel correlation.'
        return
    elif binfactor < 1:
        print 'ERROR: Please specify a valid binning factor. Value should be a positive integer and values greater than 4 will result in data being treated as having no channel correlation.'
        return

    if outfile:
        if not ((type(outfile) is str) or (type(outfile) is list)):
            print "ERROR: Please supply a valid outfile path or list of paths (matching the number of filter images)."
            return

    # parse whether we have a bank of filters or single filter and check that number of outfiles matches
    if type(filterfile) is list:
        multifilter = True
        nfilter = len(filterfile)
        if outfile:
            if len(outfile) != len(filterfile):
                print "ERROR: Number of filter images must match the number of outfile paths."
                return
    else:
        multifilter = False



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

    if len(data.wgts.shape) > 2:
        data.wgts = np.squeeze(data.wgts)

    wgt_dims = len(data.wgts.shape)

    if wgt_dims == 2:
        print "Dataset has a weight spectrum, compressing channelized weights via averaging to a single weight per visibility."
        data.wgts = np.mean(data.wgts, axis=1)

    if weights == 'statwt':
        data.wgts *= 0.5
    elif weights == 'preserve':
        print "Assuming data weights are correct as-is. If resulting spectrum is not properly normalized, consider using 'renormalize' or applying statwt to the data."
    else:
        # using weight value as a temporary sketchy replacement for finding flagged visibilities
        wgt_mean = np.mean(data.wgts[data.wgts > 0.00001])
        data_std = np.std(data.VV[data.wgts > 0.00001])
        data.wgts *= (1/data_std**2)/wgt_mean

    # check if weights look correct
    wgt_mean = np.mean(data.wgts[data.wgts > 0.00001])
    data_std = np.std(data.VV[data.wgts > 0.00001])
    
    weight_offset = np.abs(wgt_mean - 1/data_std**2)/wgt_mean*100

    if weight_offset > 25.:
        print "WARNING: data weights are more than 25% offset that expected from the total data variance. This may be due to very strong lines in the data or improperly initialized data weights. If resulting spectrum is not properly normalized, consider using 'renormalize' or applying statwt to the data."

    # check to see if binfactor is 1. if so, bin by a factor of 2 as covariance matrix of unbinned data is ill-conditioned
    if binfactor == 1 and window_func == "Hanning":
        print "WARNING: unbinned Hanning smoothed data has an ill-conditioned covariance matrix. Binning data by a factor of 2 and adjusting weights to keep numerically stable. Note that channel numbers in the output filter response will correspond to the binned data. Frequencies or velocities (if selected as output mode) will be properly calculated for the binned data."
        # force the data to have an even number of channels
        if data.VV.shape[1] & 0x1:
            data.VV = data.VV[:,:-1]
            data.freqs = data.freqs[:-1]

        data.VV = data.VV.reshape(nvis, data.VV.shape[1]/2, 2).mean(axis=2)
        data.freqs = np.ndarray.tolist(np.array(data.freqs).reshape(data.VV.shape[1], 2).mean(axis=1))
        data.wgts *= 5./3.

    if verbose: 
        t1 = time.time()
        print "Read data file: "+datafile
        print "Data read time = " + str(t1-t0)
        


    ##########################################   
    ##########################################
    #######  Single filter image case  #######
    ##########################################
    ##########################################

    if multifilter == False:

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
            print "Read filter image: " + datafile
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



        #########################################
        #   Calculate the filter visibilities   #
        #########################################

        if verbose:
            print "Generating kernel"
            t0 = time.time()

        nchan_kernel = len(filter_img.freqs)

        kernel = np.empty(nchan_kernel*nvis, dtype='complex128').reshape(nvis, nchan_kernel)
        kernel[:,:] = vis_sample(imagefile=filter_img, uu=data.uu, vv=data.vv, mu_RA=mu_RA, mu_DEC=mu_DEC)

        # calculate the noise covariance matrix and its inverse
        if window_func == "none":
            R_inv = np.identity(nchan_kernel)

        else:
            # now we assuming window_func is "Hanning"
            if binfactor > 4:
                # treat binning factors larger than 4 as having no channel correlation (valid for Hanning window function)
                R_inv = np.identity(nchan_kernel)

            elif (binfactor == 1) or (binfactor == 2):
                diagonals = [3./10.*np.ones(1000-1), np.ones(1000), 3./10.*np.ones(1000-1)]     
                R = sparse.diags(diagonals, [-1, 0, 1], format='csc').toarray()
                R_inv = np.linalg.inv(R)[500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.)), 500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.))]

            elif binfactor == 3:
                diagonals = [1./6.*np.ones(1000-1), np.ones(1000), 1./6.*np.ones(1000-1)]          
                R = sparse.diags(diagonals, [-1, 0, 1], format='csc').toarray()
                R_inv = np.linalg.inv(R)[500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.)), 500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.))]

            elif binfactor == 4:
                diagonals = [3./26.*np.ones(1000-1), np.ones(1000), 3./26.*np.ones(1000-1)]     
                R = sparse.diags(diagonals, [-1, 0, 1], format='csc').toarray()
                R_inv = np.linalg.inv(R)[500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.)), 500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.))]

        if verbose: 
            t1 = time.time()
            print "Kernel generated"
            print "Kernel generation time = " + str(t1-t0)


        ###############################
        #   Do the actual filtering   #
        ###############################

        if verbose:
            print "Starting kernel convolution"
            t0 = time.time()

        xc = np.zeros((data.VV.shape[1] - nchan_kernel + 1), dtype='complex128')  
        kernel_noise_power = 0.

        for v in np.arange(nvis):
            # sketchy temporary check for flagged visibilities
            if (not np.isnan(data.wgts[v])) and (data.wgts[v] > 0.00001):
                xc += np.correlate(data.VV[v], np.matmul(data.wgts[v]*R_inv, kernel[v]))
                kernel_noise_power += np.dot(kernel[v],np.matmul(data.wgts[v]*R_inv, kernel[v].conj()))

        # normalize the output such that real and imag noise powers are both 1 (hence factor of sqrt(2))
        xc = xc/np.sqrt(kernel_noise_power)*np.sqrt(2)

        if norm_chans:
            noise_xc = []
            for i in range(len(norm_chans)):
                noise_xc.extend(xc[norm_chans[i][0]:norm_chans[i][1]])
            noise_xc = np.array(noise_xc)
            xc_real_std = np.std(np.real(noise_xc))
            xc = xc/xc_real_std

        if verbose: 
            t1 = time.time()
            print "Data filtered"
            print "Kernel convolution time = " + str(t1-t0)
            print "max signal = " + str(np.max(np.real(xc))) + " sigma"


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
            if not restfreq:
                restfreq = np.mean(data.freqs)/1.e6
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




    #########################################   
    #########################################
    #######  Multi filter image case  #######
    #########################################
    #########################################

    elif multifilter == True:
        outdata = []

        for filter_index in range(nfilter):
            curr_filterfile = filterfile[filter_index]
            if outfile:
                curr_outfile = outfile[filter_index]

            #############################
            #   Read the filter image   #
            #############################

            # now that we have the data, let's import the filter file
            if verbose:
                print "Reading filter file " + str(filter_index+1) + " of " + str(nfilter) + ": " + curr_filterfile
                t0 = time.time()

            if isinstance(curr_filterfile, SkyImage):
                filter_img = curr_filterfile
            elif "image.out" in curr_filterfile:
                if src_distance is None:
                     print "ERROR: A source distance in pc needs to be provided in order to process a RADMC3D image file"
                     return 
                else: filter_img = import_model_radmc(src_distance, curr_filterfile)
            elif "fits" in curr_filterfile:
                filter_img = import_model_fits(curr_filterfile)
            else:
                print "ERROR: Not a valid filter image option. Please provide a FITS file, a RADMC3D image file, or a SkyImage object)."
                return 

            # the number of filter channels needs to be smaller than the data channels
            if (len(filter_img.freqs) >= len(data.freqs)):
                print "ERROR: Number of channels in filter exceeds number of data channels. Filtering cannot continue."
                return

            elif (len(filter_img.freqs) >= len(data.freqs)*0.5):
                print "WARNING: Number of channels in data file seems small compared to width of filter. Make sure there is adequate baseline in the data file."

            if verbose: 
                t1 = time.time()
                print "Read filter image: " + curr_filterfile
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



            #########################################
            #   Calculate the filter visibilities   #
            #########################################

            if verbose:
                print "Generating kernel"
                t0 = time.time()

            nchan_kernel = len(filter_img.freqs)

            kernel = np.empty(nchan_kernel*nvis, dtype='complex128').reshape(nvis, nchan_kernel)
            kernel[:,:] = vis_sample(imagefile=filter_img, uu=data.uu, vv=data.vv, mu_RA=mu_RA, mu_DEC=mu_DEC)

            kernel = kernel/np.mean(np.abs(kernel))

            # calculate the noise covariance matrix and its inverse
            if window_func == "none":
                R_inv = np.identity(nchan_kernel)

            else:
                # now we assuming window_func is "Hanning"
                if binfactor > 4:
                    # treat binning factors larger than 4 as having no channel correlation (valid for Hanning window function)
                    R_inv = np.identity(nchan_kernel)

                elif (binfactor == 1) or (binfactor == 2):
                    diagonals = [3./10.*np.ones(1000-1), np.ones(1000), 3./10.*np.ones(1000-1)]     
                    R = sparse.diags(diagonals, [-1, 0, 1], format='csc').toarray()
                    R_inv = np.linalg.inv(R)[500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.)), 500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.))]

                elif binfactor == 3:
                    diagonals = [1./6.*np.ones(1000-1), np.ones(1000), 1./6.*np.ones(1000-1)]          
                    R = sparse.diags(diagonals, [-1, 0, 1], format='csc').toarray()
                    R_inv = np.linalg.inv(R)[500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.)), 500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.))]

                elif binfactor == 4:
                    diagonals = [3./26.*np.ones(1000-1), np.ones(1000), 3./26.*np.ones(1000-1)]     
                    R = sparse.diags(diagonals, [-1, 0, 1], format='csc').toarray()
                    R_inv = np.linalg.inv(R)[500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.)), 500-int(nchan_kernel/2.) : 500+int(math.ceil(nchan_kernel/2.))]

            if verbose: 
                t1 = time.time()
                print "Kernel generated"
                print "Kernel generation time = " + str(t1-t0)


            ###############################
            #   Do the actual filtering   #
            ###############################

            if verbose:
                print "Starting kernel convolution"
                t0 = time.time()

            curr_xc = np.zeros((data.VV.shape[1] - nchan_kernel + 1), dtype='complex128')  
            kernel_noise_power = 0.

            for v in np.arange(nvis):
                # sketchy temporary check for flagged visibilities
                if (not np.isnan(data.wgts[v])) and (data.wgts[v] > 0.00001):
                    curr_xc += np.correlate(data.VV[v], np.matmul(data.wgts[v]*R_inv, kernel[v]))
                    kernel_noise_power += np.dot(kernel[v],np.matmul(data.wgts[v]*R_inv, kernel[v].conj()))

            # normalize the output such that real and imag noise powers are both 1 (hence factor of sqrt(2))
            curr_xc = curr_xc/np.sqrt(kernel_noise_power)*np.sqrt(2)

            if norm_chans:
                curr_noise_xc = []
                for i in range(len(norm_chans)):
                    curr_noise_xc.extend(curr_xc[norm_chans[i][0]:norm_chans[i][1]])
                curr_noise_xc = np.array(curr_noise_xc)
                curr_xc_real_std = np.std(np.real(curr_noise_xc))
                curr_xc = curr_xc/curr_xc_real_std

            if verbose: 
                t1 = time.time()
                print "Data filtered"
                print "Kernel convolution time = " + str(t1-t0)
                print "max signal = " + str(np.max(np.real(curr_xc))) + " sigma"


            ###############################
            #   Calculate output x-axis   #
            ###############################

            if mode=='channel':
                response_chans = np.arange(curr_xc.shape[0]) + nchan_kernel/2 + 0.5
                curr_x_axis = response_chans

            elif mode=='frequency':
                response_freqs = (np.squeeze(data.freqs[nchan_kernel/2:-nchan_kernel/2+1]) + data_delfreq/2.0)/1.e6
                curr_x_axis = response_freqs

            else:
                if not restfreq:
                    restfreq = np.mean(data.freqs)/1.e6
                response_freqs = (np.squeeze(data.freqs[nchan_kernel/2:-nchan_kernel/2+1]) + data_delfreq/2.0)/1.e6
                response_vels = (restfreq - response_freqs)/restfreq*c_kms
                curr_x_axis = response_vels




            ############################
            #   Plot filter response   #
            ############################

            if plot==True:

                fig = pl.figure(figsize=(5,2.7), dpi=300)
                ax = pl.axes([0.12,0.13,0.85,0.84])

                if mode=='channel':
                    pl.plot(response_chans, np.real(curr_xc))

                elif mode=='frequency':
                    pl.plot(response_freqs, np.real(curr_xc))

                else:
                    pl.plot(response_vels, np.real(curr_xc))

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
                np.save(curr_outfile, np.vstack((curr_x_axis, curr_xc)))    
            
            # otherwise we're going to return the raw output of the filtering
            else:
                outdata.append(np.vstack((curr_x_axis, curr_xc)))

        if outfile:
            return
        else:
            return outdata
    return
