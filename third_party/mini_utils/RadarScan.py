import numpy as np
import h5py

from PIL import Image
from skimage import measure
from skimage.measure import block_reduce
from skimage.util.shape import view_as_windows
from scipy.interpolate import griddata
# from mpl_toolkits.basemap import Basemap
from matplotlib.figure import Figure
from matplotlib import cm
import matplotlib.pyplot as plt
import xarray as xr


class RadarScan:
    # Constructor
    def __init__(self, file_path, elevation, size=50000, site='meron', VRAD_filter=True, CLOUDS_filter=False, data_type='VRAD', 
                    pixelResolution=256, interpolation='nearest', projType='cartesian', GCS=False, cutV=None, 
                        MIN_INTERP_PX=5, **kwargs):
        
        # Read the hdf file
        self.hdfFile = hdfReader(file_path, site=kwargs.get('site', site))

        # The elevation of the scan
        self.elevation = elevation
        
        # Set image properties
        self.size = kwargs.get('size', size)  # The size of the bounding box of the image
        self.pixelResolution = kwargs.get('pixelResolution', pixelResolution)  # Resolution of resulting image
        self.interpolation = kwargs.get('interpolation', interpolation)  # Type of interpolation to use when resizing image
        self.projType = kwargs.get('projType', projType)
        self.MIN_INTERP_PX = kwargs.get('MIN_INTERP_PX', MIN_INTERP_PX)

        # Set data properties
        self.data_type = kwargs.get('data_type', data_type)  # The type of the parameter to extract
        self.VRAD_filter = kwargs.get('VRAD_filter', VRAD_filter)  # Filter the VRAD or not
        self.CLOUDS_filter = kwargs.get('CLOUDS_filter', CLOUDS_filter)  # Filter clouds or not
        self.GCS = kwargs.get('GCS', GCS)
        
        # Setting polar array
        self.polarData, self.T, self.R = self.getPolarParamData()

        # Change pixel resolution if full resolution is needed
        if self.interpolation == 'full':
            self.pixelResolution = 2 * self.R.shape[0] - 1

        # Setting cartesian array
        if self.projType == 'cartesian':
            # Converting polar data to cartesian
            self.X, self.Y = self.getCartesCoords()
            self.cartesData = self.polar2Cartes()
            # Convert cartesian coordinates to lat lon
            if self.GCS:
                self.cartes2latlon()
            if cutV is not None:
                if cutV[0] == 'up':
                    self.cutUpVRAD(cutV[1])
                elif cutV[0] == 'down':
                    self.cutDownVRAD(cutV[1])
                else:
                    print("ERROR IN cutV")
                
                
                
                
    # Convert cartesian coordinates to GCS (Geographic Coordinate System) - lat, lon
    def cartes2latlon(self, left = 34.872396, right = 35.944386, bottom = 32.5515189, top = 33.4505489):
        # radar bounding box in lat lon coordinates
        coordBox = [left, right, bottom, top]

        # Change coordinates to lat, lon coordinates
        self.X = np.linspace(coordBox[0], coordBox[1], self.X.shape[0])
        self.Y = np.linspace(coordBox[2], coordBox[3], self.Y.shape[0])
        
        # Change coordinates in the data array
        self.cartesData = self.createDataArray(self.cartesData.data, self.X, self.Y)
        
        
        
    # Get cartesian data array
    def getPPI(self):
        return self.cartesData

    
    # Get polar xarray
    def getPolarXr(self):
        return xr.DataArray(self.polarData, coords={'T': self.T, 'R': self.R})
    
    
    
    
    def getPolarParamData(self):
        # Get polar VRAD data from HDF
        polarData = self.hdfFile.getData(self.data_type, self.elevation)
        # Process the data
        self.processData(polarData)
        return polarData.data, polarData.T, polarData.R
    
    def processData(self, param):
        # Cut array
        self.cutRange(param)
        # Filter VRAD
        if self.VRAD_filter and self.data_type == 'VRAD':
            param.filterData()
        # filter clouds
        if self.CLOUDS_filter and self.data_type == 'VRAD':
            with h5py.File(self.hdfFile.filePath, 'r') as f:   
                param.data = self.cloudFilter(f, param.data)
            
    
    def cutRange(self, param):
        # Get max range bins
        maxRangeBins = self.getMaxRangeBins(param)
        param.cutRange(maxRangeBins)
    
    def getMaxRangeBins(self, param):
        return int(2**0.5 * self.size/param.r_res)
      
            
            
            
            
    def polar2Cartes(self):
        # Interpolate data from polar to cartesian
        cartesData = self.interpPolar2Cartes()
        # Resize data if interpolation is average
        cartesData = self.resizeCartes(cartesData)
        # Make xarray data array from data
        cartesDataX = self.createDataArray(cartesData, self.X, self.Y)
        # Set bounding box for the data
        cartesDataX = self.boxCartesXR(cartesDataX)
        return cartesDataX
        
    def interpPolar2Cartes(self):
        # Create cartesian grid of polar coords
        PXX, PYY = self.getPolar2CartesGrid()
        # Create cartesian grid for interpolation
        XX, YY = self.getCartesGrid()
        # Interpolate missing cartesian coords
        return self.interpFunc(PXX, PYY, XX, YY)
    
    def resizeCartes(self, cartesData):
        if self.interpolation == 'average':
            cartesData = self.strided_conv_avg(cartesData)
        elif self.interpolation == 'median':
            cartesData = self.strided_conv_median(cartesData)
        self.X, self.Y = self.getCartesCoords(interpType='nearest')
        return cartesData
    
    
    def interpFunc(self, PXX, PYY, XX, YY):
        return griddata(
            (PXX.flatten(), PYY.flatten()),
            self.polarData.flatten(),
            (XX, YY),
            method='nearest'
        )
            
    def getPolar2CartesGrid(self):
        # create a polar grid
        RR, TT = np.meshgrid(self.R, self.T)
        # convert to cartesian grid
        PXX = RR * np.cos(np.pi/2 - TT)
        PYY = RR * np.sin(np.pi/2 - TT)
        return PXX, PYY 
    
    def getCartesGrid(self):
        return np.meshgrid(self.X, self.Y)
    
    def getCartesCoords(self, interpType=None):
        if interpType is None:
            interpType = self.interpolation
        if interpType == 'nearest' or interpType == 'full':
            X = np.linspace(-np.max(self.R), np.max(self.R), int(2**0.5*self.pixelResolution))
            Y = np.linspace(-np.max(self.R), np.max(self.R), int(2**0.5*self.pixelResolution))
        elif interpType == 'average' or interpType == 'median':
            diameterPxLen = 2 * self.R.shape[0] - 1
            X = np.linspace(-np.max(self.R), np.max(self.R), diameterPxLen)
            Y = np.linspace(-np.max(self.R), np.max(self.R), diameterPxLen)
        else:
            raise ValueError("wrong interpolation")
        return X, Y
    
    def strided_conv_avg(self, image_n):
        # set the desired size
        new_size_n = (int(2**0.5*self.pixelResolution), int(2**0.5*self.pixelResolution))
        # Copy image and make all nans 0
        image = image_n.copy()
        image[np.isnan(image)] = 0
        # Create array to count non 0's
        positivesArr = np.zeros_like(image)
        positivesArr[image > 0] = 1
        # Create array to count 0's
        zerosArr = np.zeros_like(image)
        zerosArr[image == 0] = 1
        
        # Define kernel
        kernel_step = int(image.shape[0]/new_size_n[0])
        kernel = np.ones((kernel_step, kernel_step))
        # Create kernel to find 0's in new array
        zeros_kernel = np.zeros((kernel_step, kernel_step))
        zeros_kernel[int(kernel_step/2), int(kernel_step/2)] = 1
        
        # Make arrays for strided covolution
        image4D = view_as_windows(image, kernel.shape, step=kernel_step)
        positivesArr4D = view_as_windows(positivesArr, kernel.shape, step=kernel_step)
        # zerosArr4D = view_as_windows(zerosArr, kernel.shape, step=kernel_step)
        # Apply convolution
        sumArr = np.tensordot(image4D, kernel, axes=((2,3),(0,1)))
        countArr = np.tensordot(positivesArr4D, kernel, axes=((2,3),(0,1)))
        # zeroLocArr = np.tensordot(image4D, zeros_kernel, axes=((2,3),(0,1)))

        # Calculate average values of non 0 values
        avgArr = np.divide(sumArr, countArr, out=np.zeros_like(sumArr), where=countArr != 0)
        # avgArr = sumArr/np.max(sumArr) * 255
        
        # Where there are values which are taken only from the sides of the kernel, set value to 0
        # avgArr[zeroLocArr == 0] = 0
        
        # Where there are too little pixels, remove the pixel - controlling the amount of detail we want
        avgArr[countArr < self.MIN_INTERP_PX] = 0
        
        # Get 0's back to nans
        avgArr[avgArr == 0] = np.nan

        # Correct image resolution if resolutions are not multiples of eachother
        return np.array(Image.fromarray(avgArr).resize((new_size_n), Image.NEAREST))
    
    
    def strided_conv_median(self, image_n):
        # set the desired size
        new_size_n = (int(2**0.5*self.pixelResolution), int(2**0.5*self.pixelResolution))
        # Copy image
        image = image_n.copy()
        
        # Define kernel
        kernel_step = int(image.shape[0]/new_size_n[0])
        kernel = (kernel_step, kernel_step)
        
        # Convert the image to a tensor
        image = image[np.newaxis, np.newaxis, ...]
        image_tensor = torch.from_numpy(image)
        
        # Filter r
        
        # Create median layer
        layer=MedianPool2d(kernel, kernel_step)
        # Apply median layer on image
        output = np.array(layer.forward(image_tensor)).squeeze()

        return np.array(Image.fromarray(output).resize((new_size_n), Image.NEAREST))

    
    def strided_conv_avg_new(self, image_n):
        # set the desired size
        new_size_n = (int(2**0.5*self.pixelResolution), int(2**0.5*self.pixelResolution))
        
        image = image_n.copy()
        image[np.isnan(image)] = 0
        kernel_step = int(image.shape[0]/new_size_n[0])
        
        block_size = (kernel_step, kernel_step)
        avgArr = block_reduce(image, block_size=block_size, func=np.max)
        # avgArr = sumArr/np.max(sumArr) * 255
        # avgArr = sumArr
        
        # Where there are 0's in the original array, set vlaue to 0
        # avgArr[ = 
        
        avgArr[avgArr == 0] = np.nan
        return np.array(Image.fromarray(avgArr).resize((new_size_n), Image.NEAREST))
    
    def get_xmin(self):
        return -1 * int(self.size)
    def get_xmax(self):
        return int(self.size)
    def get_ymin(self):
        return -1 * int(self.size)
    def get_ymax(self):
        return int(self.size)
    

    def boxCartesXR(self, da):
        # Bound the coordinates to the box
        center = self.X.shape[0]//2
        startIndex = center - self.pixelResolution//2 - 1
        endIndex = center + self.pixelResolution//2 + self.pixelResolution%2 - 1
        self.X = self.X[startIndex:endIndex]
        self.Y = self.Y[startIndex:endIndex]
        # Bound the data to the box
        return da.isel(x=slice(startIndex, endIndex), y=slice(startIndex, endIndex))
    
    
    def createDataArray(self, data, x, y):
        return xr.DataArray(data, coords={'y': y, 'x': x})
    
    
    def showPPI(self):
        fig, ax = plt.subplots(dpi=200)
        ax.set_facecolor('grey')
        self.initialize_grid(fig, ax)
    
    def initialize_grid(self, fig, ax):
        # get initial data
        da = self.getPPI()
        # initialize mesh grid
        mesh = da.plot.pcolormesh(ax=ax, add_colorbar=False)

        # set bounding box
        ax.set_xlim(self.get_xmin(), self.get_xmax())
        ax.set_ylim(self.get_ymin(), self.get_ymax())

        # Remove white space
        ax.set_xticks([])
        ax.set_yticks([])

        # make image rectangular
        ax.set_aspect('equal')

        # remove labels
        ax.set_xlabel('')
        ax.set_ylabel('')
        txt = ax.text(-45000, 45000, str(get_time(self.hdfFile.filePath).strftime("%Y%m%d %H%M%S")), fontsize=6, color='black', fontweight='bold')
        return mesh, txt
    
    
    def update(self):
        self.X, self.Y = self.getCartesCoords()
        self.cartesData = self.polar2Cartes()
        # Convert cartesian coordinates to lat lon
        if self.GCS:
            self.cartes2latlon()
            
    def plot(self, y1=32.36797, y2=33.4679789, x1=34.85628, x2=35.9562869, cbar=False, figsize=(8,8)):
        fig, ax = self.createFig(figsize)
        self.cartesData.plot.imshow(ax=ax, zorder=2, add_colorbar=cbar)
        if self.GCS:
            ax.set_xlim(x1, x2)
            ax.set_ylim(y1, y2)
        return fig
    
    def save(self, filename):
        array = self.cartesData.values
        array = np.flipud(array)
        array = array / 255.0

        viridis_cmap = cm.get_cmap('viridis')
        norm = plt.Normalize(vmin=np.nanmin(array), vmax=np.nanmax(array))
        
        viridis_array = viridis_cmap(norm(array))
        viridis_array = viridis_array * 255
        viridis_array[viridis_array[:,:,3] == 0] = 127
        viridis_array = viridis_array[:,:,0:3]
        
        viridis_image = Image.fromarray((viridis_array).astype(np.uint8))
        viridis_image.save(filename)
        return viridis_array
    
    def createFig(self, figsize, detailed=False):
        if self.GCS:
            fig = Figure(figsize=figsize)
            ax = fig.add_subplot(111)
            m = Basemap(llcrnrlon=34.028356, llcrnrlat=29.4394549,
                        urcrnrlon=35.9562869, urcrnrlat=33.4679789,
                        resolution='i',  # Set using letters, e.g. c is a crude drawing, f is a full detailed drawing
                        projection='merc',  # The projection style is what gives us a 2D view of the world for this
                        lon_0=34.7818, lat_0=32.0853,  # Setting the central point of the image
                        epsg=4326,
                        ax=ax)  # Setting the coordinate system we're using
            if detailed:
                ax.imshow(Image.open(r'C:\Users\asdds\Documents\ilya\work\Knafei_Silon\UNET-flocks-detection-main\aip_Basic_Map (1)\map\as.jpg'), extent=[34.028356, 35.9562869, 29.4394549, 33.4679789], zorder=2)
            else:
                m.drawmapboundary(fill_color='#82bfed')  # Make your map into any style you like
                m.fillcontinents(color='#bfb375', lake_color='#82bfed')  # Make your map into any style you like
                m.drawcoastlines()
                m.drawrivers()  # Default colour is black but it can be customised
                m.drawcountries()
        else:
            fig, ax = plt.subplots(dpi=300)
        return fig, ax
    
    
    def cutBox(self, x1, x2, y1, y2):
        self.cartesData = self.cartesData.sel(x=slice(x1, x2),y=slice(y1, y2))
        
    
    def getValues(self):
        o = self.hdfFile.getData(self.data_type, self.elevation).offset
        g = self.hdfFile.getData(self.data_type, self.elevation).gain
        return self.cartesData * g + o
    
    
    def cutDownVRAD(self, cutV):
        self.cartesData.data[self.getValues().data > cutV] = np.nan
    def cutUpVRAD(self, cutV):
        self.cartesData.data[self.getValues().data < cutV] = np.nan
    
    def find_best_divisor(self, size, low, high, step=1):
        minimal_truncation, best_divisor = min((size % divisor, divisor)
            for divisor in range(low, high, step))
        return best_divisor

    
#     # Takes dBz and VRAD and filters VRAD where dBz has "cloud" characteristics
#     # dBz has to be after conversion:  dbz = Gain*dbz + offset
#     def cloudFilter(self, f, vrad, target='VRAD', MIN_DBZ=0, MIN_COMP_SIZE=5, MIN_DBZ_MEAN=-10):
#         # take vrad
#         vrad_temp = vrad.copy()
#         zeros_mask_v = vrad_temp == 0

#         # take dbz
#         dbz_param = self.hdfFile.getData('DBZ', self.elevation)
#         self.cutRange(dbz_param)
#         dbz_temp = dbz_param.data.copy()
#         zeros_mask_z = dbz_temp == 0
#         dbz = dbz_param.getDataVal()
#         # zeros_mask_z[dbz < MIN_DBZ_MEAN] = True
#         dbz_temp[zeros_mask_z] = 0

#         # binarize data
#         v = vrad_temp.copy()
#         v[v > np.min(v)] = np.max(v)

#         z = dbz_temp.copy()
#         z[z > np.min(z)] = np.max(z)

#         # find connected components
#         labeled_v, num_labels = measure.label(v, connectivity=2, background=0, return_num=True)
#         labeled_z, num_labels = measure.label(z, connectivity=2, background=0, return_num=True)

#         # find coordinates of very high dbz
#         dbz_coords = np.where(dbz > MIN_DBZ)

#         for i, j in zip(*dbz_coords):  # if there is a high dbz point there
#             label_v = labeled_v[i, j]  # find connected component of the high dbz in VRAD
#             label_z = labeled_z[i, j]  # find connected component of the high dbz in dBz
#             if label_v > 0:  # don't touch background component
#                 mask_v = labeled_v == label_v  # create masks for the components
#                 mask_z = labeled_z == label_z
#                 if np.sum(mask_v) > MIN_COMP_SIZE:  # don't touch small connected components
#                     if np.mean(dbz[mask_z]) > MIN_DBZ_MEAN:  # don't touch low dBz components, to not filter birds
#                         # vrad_temp[mask_v] = np.max(vrad_temp)
#                         if 'VRAD' in target:
#                             zeros_mask_v[mask_z] = True
#                         if 'DBZ' in target:
#                             zeros_mask_z[mask_z] = True  # remove the "cloud"
#                 labeled_v[mask_v] = 0  # don't waste time on labels already checked
#         if 'VRAD' in target:
#             vrad_temp[zeros_mask_v] = 0
#             return vrad_temp
#         if 'DBZ' in target:
#             dbz_temp[zeros_mask_z] = 0
#             return dbz_temp 
    
    # Takes dBz and VRAD and filters VRAD where dBz has "cloud" characteristics
    # dBz has to be after conversion:  dbz = Gain*dbz + offset
    def cloudFilter(self, f, arr, target='VRAD', MIN_DBZ=0, MIN_COMP_SIZE=5, MIN_DBZ_MEAN=-10):
        # take vrad
        vrad_temp = arr.copy()
        zeros_mask_v = vrad_temp == 0

        # take dbz
        dbz_param = self.hdfFile.getData('DBZ', self.elevation)
        self.cutRange(dbz_param)
        dbz_temp = dbz_param.data.copy()
        zeros_mask_z = dbz_temp == 0
        dbz = dbz_param.getDataVal()
        # zeros_mask_z[dbz < MIN_DBZ_MEAN] = True
        dbz_temp[zeros_mask_z] = 0

        # binarize data
        v = vrad_temp.copy()
        v[v > np.min(v)] = np.max(v)

        z = dbz_temp.copy()
        z[z > np.min(z)] = np.max(z)

        # find connected components
        labeled_v, num_labels = measure.label(v, connectivity=2, background=0, return_num=True)
        labeled_z, num_labels = measure.label(z, connectivity=2, background=0, return_num=True)

        # find coordinates of very high dbz
        dbz_coords = np.where(dbz > MIN_DBZ)

        for i, j in zip(*dbz_coords):  # if there is a high dbz point there
            label_v = labeled_v[i, j]  # find connected component of the high dbz in VRAD
            label_z = labeled_z[i, j]  # find connected component of the high dbz in dBz
            if label_v > 0:  # don't touch background component
                mask_v = labeled_v == label_v  # create masks for the components
                mask_z = labeled_z == label_z
                if np.sum(mask_v) > MIN_COMP_SIZE:  # don't touch small connected components
                    if np.mean(dbz[mask_z]) > MIN_DBZ_MEAN:  # don't touch low dBz components, to not filter birds
                        # vrad_temp[mask_v] = np.max(vrad_temp)
                        if 'VRAD' in target:
                            zeros_mask_v[mask_z] = True
                        if 'DBZ' in target:
                            zeros_mask_z[mask_z] = True  # remove the "cloud"
                labeled_v[mask_v] = 0  # don't waste time on labels already checked
        if 'VRAD' in target:
            vrad_temp[zeros_mask_v] = 0
            return vrad_temp
        if 'DBZ' in target:
            dbz_temp[zeros_mask_z] = 0
            return dbz_temp  
    
    
    
    
    # def get_pred_ds(self, pred):
        
    

        
    

        
    

        
    
class hdfReader:
    # Constructor
    def __init__(self, filePath, site='meron'):
        # The path to the hdf
        self.filePath = filePath
        self.site = site
        # Set non-situational attributes - which are common to all parameters
        with h5py.File(self.filePath, 'r') as f:            
            self.PARAMS = self.getParamNames(f)

            
            
            
    # Returns an hdfParam object of the parameter "param" ('VRAD', 'DBZ', 'SQI'...) at elevation elev
    # See showParams() to see which parameters are available
    def getData(self, param: str, elev):
        with h5py.File(self.filePath, 'r') as f:
            # Find the desired parameter
            paramDict = [d for key, d in f[f'dataset{elev}'].items() if 'data' in key and param in str(d['what'].attrs['quantity'])][0]
            
            # Get the information about the parameter
            data = self.getParamData(paramDict)
            name = self.getParamName(paramDict)
            gain = self.getParamGain(paramDict)
            offset = self.getParamOffset(paramDict)
            
            # Azimuths
            T = self.getAzimuths(f, elev)
            # ranges
            R = self.getRanges(f, elev)
            # Range resolution
            r_res = self.getRangeResolution(f, elev)
            
            # Return an hdfParam object
            return hdfParam(data, T, R, r_res, name, gain, offset)
    
    
    
    
    def getParamData(self, paramDict):
        tempData = paramDict['data'][()].copy()
        if self.site == 'meron':
            tempData = np.roll(tempData, 145, axis=0)
        return tempData
    
    def getParamName(self, paramDict):
        return paramDict['what'].attrs['quantity']
    
    def getParamGain(self, paramDict):
        return paramDict['what'].attrs['gain']
    
    def getParamOffset(self, paramDict):
        return paramDict['what'].attrs['offset']
    
    def getParamNames(self, f, elev=1):
        return [d['what'].attrs['quantity'] for key, d in f[f'dataset{elev}'].items() if 'data' in key]
    
    def getAzimuths(self, f, elev):
        t_bins = f[f'dataset{elev}']['where'].attrs['nrays']
        t_res = 360/t_bins
        return np.linspace(0, 2 * np.pi, t_bins, endpoint=False)
    
    def getRanges(self, f, elev):
        r_res = f[f'dataset{elev}']['where'].attrs['rscale']
        r_bins = f[f'dataset{elev}']['where'].attrs['nbins']
        return np.linspace(0, r_res * r_bins, r_bins, endpoint=False)
    
    def getRangeResolution(self, f, elev):
        return f[f'dataset{elev}']['where'].attrs['rscale']
    
    def showParams(self):
        print(self.PARAMS)
        

class hdfParam:
    # Constructor
    def __init__(self, data, T, R, r_res, paramName, gain, offset, nanBackground=True):
        self.nanBackground = nanBackground
        # Name of the parameter
        self.paramName = paramName
        # Range resolution
        self.r_res = r_res
        # Ranges
        self.R = R
        # Azimuths
        self.T = T
        # Data in uint8, 0-255
        self.data = self.setData(data)
        # Gain of the parameter
        self.gain = gain
        # Offset of the parameter
        self.offset = offset  
        
        

        
        
    # Filter parameter data using thresholds
    def filterData(self, minThreshold=-20, maxThreshold=20):
        # Get actual values of the parameter
        dataVal = self.getDataVal()
        # Treshold the valus
        self.thresholdVal(dataVal, minThreshold, maxThreshold)
    
    def thresholdVal(self, dataVal, minThreshold, maxThreshold):
        if self.nanBackground:
            self.data[dataVal < minThreshold] = np.nan
            self.data[dataVal > maxThreshold] = np.nan
        else:
            self.data[dataVal < minThreshold] = 0
            self.data[dataVal > maxThreshold] = 0
        
    def getDataVal(self):
        return self.gain * self.data + self.offset
    
    
    
    
    def cutRange(self, maxR):
        self.data = self.data[:, :maxR]
        self.R = self.R[:maxR]
        
        
        
        
    def setData(self, data):
        tempData = data.astype(float)
        if self.nanBackground:
            tempData[tempData == 0] = np.nan
        return tempData
    