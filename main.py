import os
import pandas as pd
import tempfile
import shutil

# PARAMETERS
CSV_OUT = ".\\OUT\\trees.csv"
FLOW_OUT = ".\\OUT\\flow.png"
ROAD_OUT = ".\\OUT\\road.png"
FOREST_OUT = ".\\OUT\\forest.png"
SHRUB_OUT = ".\\OUT\\shrub.png"

TREES_IN = 'trees.shp'
ELEVATION_IN = 'elevation.tif'
ROADS_IN = 'roads.shp'

START_POINT_EAST = 421986
START_POINT_SOUTH = 4825671
WIDTH = 922.6296
HEIGHT = 922.6296

FEET_TO_METER = 0.3048
METER = False

# Creating temporary folder
temporary_folder = tempfile.TemporaryDirectory()


class Parameter:
    def __init__(self, crs, start_east, start_south, width, height):
        self.crs = crs
        self.start_east = start_east
        self.start_south = start_south
        self.width = width
        self.height = height

    def get_extent(self):
        return f'{self.start_east},{self.start_east + self.width},{self.start_south},{self.start_south + self.height} [{self.crs}]'

    def raster_calculator(self, raster_in, raster_out, formula):
        return {
            'INPUT_A': raster_in,
            'BAND_A': 1,
            'INPUT_B': None,
            'BAND_B': None,
            'INPUT_C': None,
            'BAND_C': None,
            'INPUT_D': None,
            'BAND_D': None,
            'INPUT_E': None,
            'BAND_E': None,
            'INPUT_F': None,
            'BAND_F': None,
            'FORMULA': formula,
            'NO_DATA': None,
            'EXTENT_OPT': 0,
            'PROJWIN': None,
            'RTYPE': 5,
            'OPTIONS': '',
            'EXTRA': '',
            'OUTPUT': raster_out
        }

    def flow_map(self, raster_in, raster_out, memory):
        return {
            'elevation': raster_in,
            '-s': False,
            'd8cut': None,
            'memory': memory,
            'filled': 'TEMPORARY_OUTPUT',
            'direction': 'TEMPORARY_OUTPUT',
            'swatershed': 'TEMPORARY_OUTPUT',
            'accumulation': 'TEMPORARY_OUTPUT',
            'tci': raster_out,
            'stats': 'TEMPORARY_OUTPUT',
            'GRASS_REGION_PARAMETER': None,
            'GRASS_REGION_CELLSIZE_PARAMETER': 0,
            'GRASS_RASTER_FORMAT_OPT': '',
            'GRASS_RASTER_FORMAT_META': ''
        }

    def slope(self, raster_in, raster_out):
        return {
            'INPUT': raster_in,
            'BAND': 1,
            'SCALE': 1,
            'AS_PERCENT': False,
            'COMPUTE_EDGES': False,
            'ZEVENBERGEN': False,
            'OPTIONS': '',
            'EXTRA': '',
            'OUTPUT': raster_out
        }

    def buffer(self, vector_in, vector_out, distance, segments):
        return {'INPUT': vector_in,
                'DISTANCE': distance,
                'SEGMENTS': segments,
                'END_CAP_STYLE': 0,
                'JOIN_STYLE': 0,
                'MITER_LIMIT': 2,
                'DISSOLVE': False,
                'OUTPUT': vector_out
                }

    def union(self, vector_in, vector_out):
        return {
            'INPUT': vector_in,
            'OVERLAYS': None,
            'OVERLAY_FIELDS_PREFIX': '',
            'OUTPUT': vector_out
        }

    def clip(self, raster_in, vector_in, raster_out):
        return {
            'INPUT': raster_in,
            'MASK': vector_in,
            'SOURCE_CRS': None,
            'TARGET_CRS': None,
            'TARGET_EXTENT': None,
            'NODATA': None,
            'ALPHA_BAND': False,
            'CROP_TO_CUTLINE': False,
            'KEEP_RESOLUTION': False,
            'SET_RESOLUTION': False,
            'X_RESOLUTION': None,
            'Y_RESOLUTION': None, 'MULTITHREADING': False,
            'OPTIONS': '', 'DATA_TYPE': 0, 'EXTRA': '',
            'OUTPUT': raster_out
        }

    def fill(self, raster_in, raster_out):
        return {
            'INPUT': raster_in,
            'BAND': 1,
            'FILL_VALUE': 0,
            'OUTPUT': raster_out}

    def sieve(self, raster_in, raster_out, threshold):
        return {
            'INPUT': raster_in,
            'THRESHOLD': threshold,
            'EIGHT_CONNECTEDNESS': False,
            'NO_MASK': False,
            'MASK_LAYER': None,
            'EXTRA': '',
            'OUTPUT': raster_out}

    def rasterizing(self, vector_in, vector_out, burn_value, width, height):
        return {
            'INPUT': vector_in,
            'BURN': burn_value,
            'USE_Z': False,
            'UNITS': 0,
            'WIDTH': width,
            'HEIGHT': height,
            'EXTENT': self.get_extent(),
            'NODATA': 0,
            'OPTIONS': '',
            'DATA_TYPE': 5,
            'INIT': None,
            'INVERT': False,
            'EXTRA': '',
            'OUTPUT': vector_out}

    def filter(self, raster_in_1, raster_in_2, raster_out):
        return {
            'INPUT': [raster_in_1,
                      raster_in_2],
            'REF_LAYER': raster_in_1,
            'NODATA_AS_FALSE': False,
            'NO_DATA': -9999,
            'DATA_TYPE': 5,
            'OUTPUT': raster_out}

    def to_png(self, raster_in, raster_out, param):
        return {
            'INPUT': raster_in,
            'TARGET_CRS': None,
            'NODATA': None,
            'COPY_SUBDATASETS': False,
            'OPTIONS': '',
            'EXTRA': '',
            'DATA_TYPE': param,
            'OUTPUT': raster_out
        }

    def smoothing(self, raster_input, raster_output, radius):
        return {
            'INPUT': raster_input,
            'RESULT': raster_output, 'SIGMA': 50, 'KERNEL_TYPE': 1, 'KERNEL_RADIUS': radius}

    def reproject_vector(self, input_vector, out_vector):
        return {
            'INPUT': input_vector,
            'TARGET_CRS': QgsCoordinateReferenceSystem(self.crs),
            'OPERATION': '+proj=pipeline +step +proj=unitconvert +xy_in=ft +xy_out=m +step +inv +proj=lcc +lat_0=41.75 +lon_0=-120.5 +lat_1=43 +lat_2=45.5 +x_0=399999.9999984 +y_0=0 +ellps=GRS80 +step +proj=utm +zone=10 +ellps=WGS84',
            'OUTPUT': out_vector
        }

    def reproject_raster(self, input_raster, out_raster):
        return {
            'INPUT': input_raster,
            'SOURCE_CRS': None,
            'TARGET_CRS': QgsCoordinateReferenceSystem(self.crs),
            'RESAMPLING': 0,
            'NODATA': None,
            'TARGET_RESOLUTION': None,
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'TARGET_EXTENT': None,
            'TARGET_EXTENT_CRS': None,
            'MULTITHREADING': False,
            'EXTRA': '',
            'OUTPUT': out_raster
        }

    def clip_vector(self, vector_in, vector_out, clip=True):
        return {
            'INPUT': vector_in,
            'EXTENT': self.get_extent(),
            'CLIP': clip,
            'OUTPUT': vector_out
        }

    def clip_raster(self, raster_in, raster_out):
        return {
            'INPUT': raster_in,
            'PROJWIN': self.get_extent(),
            'OVERCRS': False,
            'NODATA': None,
            'OPTIONS': '',
            'DATA_TYPE': 0,
            'EXTRA': '',
            'OUTPUT': raster_out
        }

    def set_m(self, vector_in, raster_in, vector_out):
        return {
            "BAND": 1,
            "INPUT": vector_in,
            "NODATA": 0.0,
            "OFFSET": 0.0,
            "OUTPUT": vector_out,
            "RASTER": raster_in,
            "SCALE": 1.0
        }

    def add_xy(self, vector_in, vector_out):
        return {
            'INPUT': vector_in,
            'CRS': QgsCoordinateReferenceSystem(self.crs),
            'PREFIX': '',
            'OUTPUT': vector_out}

    def add_xyz(self, vector_in, vector_out):
        return {
            'INPUT': vector_in,
            'SUMMARIES': [0],
            'COLUMN_PREFIX': 'm_',
            'OUTPUT': vector_out}

    def dissolve(self, vector_in, vector_out):
        return {
            'INPUT': vector_in,
            'FIELD': [],
            'SEPARATE_DISJOINT': False,
            'OUTPUT': vector_out
        }


class Helper:
    @staticmethod
    def check_folder(out_folder):
        """
        This method ensures that the given folder exists
        :param out_folder: The string of the path (absolute or relative)
        :return: Nothing
        """
        try:
            print(f"Checking {out_folder}")
            file_name = out_folder.split("\\")[-1]
            out_folder = out_folder.replace("\\" + file_name, "")
            os.makedirs(out_folder)
            print(f"Checking {out_folder}")
        except:
            print(f"Could not create Folder. {out_folder} probably already exists")
            pass


class Processor:
    def __init__(self, csv_out, flow_out, road_out, forest_out, shrub_out, start_east, start_south, width, height,
                 trees_in, elevation_in, roads_in):
        print("Checking Output-Folder")
        Helper.check_folder(csv_out)
        self.csv_out = csv_out

        Helper.check_folder(flow_out)
        self.flow_out = flow_out

        Helper.check_folder(road_out)
        self.road_out = road_out

        Helper.check_folder(forest_out)
        self.forest_out = forest_out

        Helper.check_folder(shrub_out)
        self.shrub_out = shrub_out

        self.param = Parameter("EPSG:32610", start_east, start_south, width, height)

        self.trees_in, self.road_in, self.elevation_in = self.prepare_layers(trees_in, elevation_in, roads_in)

    def save_raster(self, in_path, out_path):
        processing.runAndLoadResults("gdal:translate", self.param.to_png(in_path, out_path, 1))
        print(f"Saved to {out_path}")

    def reproject_vector(self, input_vector, out_vector):
        print(f"Re-Projecting {input_vector}")
        processing.run("native:reprojectlayer", self.param.reproject_vector(input_vector, out_vector))
        print(f"Reprojected Layer written to {out_vector}")
        return out_vector

    def reproject_raster(self, input_raster, out_raster):
        print(f"Re-Projecting {input_raster}")
        processing.run("gdal:warpreproject", self.param.reproject_raster(input_raster, out_raster))
        print(f"Reprojected Layer written to {out_raster}")

    def prepare_layers(self, trees_in, elevation_in, roads_in):
        trees_reprojected = f'{temporary_folder.name}\\trees_reprojected.shp'
        roads_reprojected = f'{temporary_folder.name}\\roads_reprojected.shp'
        elevation_reprojected = f'{temporary_folder.name}\\elevation_reprojected.tif'

        self.reproject_vector(trees_in, trees_reprojected)
        self.reproject_vector(roads_in, roads_reprojected)
        self.reproject_raster(elevation_in, elevation_reprojected)

        trees_extent = f'{temporary_folder.name}\\trees_extent.shp'
        elevation_extent = f'{temporary_folder.name}\\elevation_extent.tif'
        road_extent = f'{temporary_folder.name}\\road_extent.shp'

        self.clip_vector(trees_reprojected, trees_extent)
        self.clip_raster(elevation_reprojected, elevation_extent)
        self.clip_vector(roads_reprojected, road_extent)

        return trees_extent, road_extent, elevation_extent

    def clip_vector(self, vector_in, vector_out):
        print(f"Clipping Features: {vector_in}")
        processing.run("native:extractbyextent", self.param.clip_vector(vector_in, vector_out))
        print(f"Features saved to: {vector_out}")

    def convert(self, raster_in, raster_out):
        if not METER:
            multiplication = FEET_TO_METER
        else:
            multiplication = 1.0

        print("Converting to Meters")
        processing.run('gdal:rastercalculator',
                       self.param.raster_calculator(raster_in, raster_out, f'(A * {multiplication})'))
        print(f"Elevation saved to: {raster_out}")

    def clip_raster(self, elevation_in, elevation_extent):
        elevation_tmp = f'{temporary_folder.name}\\elevation_tmp.tif'
        print(f"Clipping Features: {elevation_in}")
        processing.run("gdal:cliprasterbyextent", self.param.clip_raster(elevation_in, elevation_extent))
        print(f"Elevation saved to: {elevation_tmp}")

    def get_m_value(self):
        vector_out = f'{temporary_folder.name}\\trees_tmp.shp'
        print("Assigning M/Z-Values")
        processing.run("native:setmfromraster", self.param.set_m(self.trees_in, self.elevation_in, vector_out))
        print(f"Assigning values completed ({vector_out})")
        return vector_out

    def save_to_csv(self, vector_in):
        trees_xy = f'{temporary_folder.name}\\trees_xy.shp'
        trees_xyz = f'{temporary_folder.name}\\trees_xyz.shp'
        print("Saving values to disk")
        data = []

        processing.run("native:addxyfields", self.param.add_xy(vector_in, trees_xy))
        processing.run("native:extractmvalues", self.param.add_xyz(trees_xy, trees_xyz))

        layer = QgsVectorLayer(trees_xyz, 'borders', 'ogr')
        if not layer.isValid():
            raise Exception('Layer is not valid')

        features = layer.getFeatures()

        for feature in features:
            data.append([feature['StandID'], feature['TreeID'], feature['Ht'], feature['CanopyArea'], feature['DBH'],
                         feature['species'], feature['x'], feature['y'], feature['m_first']])

        df = pd.DataFrame(data, columns=['StandId', 'TreeId', 'Ht', 'CanopyArea', 'DBH', 'Species', 'X', 'Y', 'Z'])

        df.to_csv(self.csv_out)
        print(f"Features saved to disk {self.csv_out}")

    def flow_accumulation(self):
        flow_unfiltered = f'{temporary_folder.name}\\flow.tif'
        flow_geotiff = f'{temporary_folder.name}\\flow_geotiff.tif'
        flow_scaled = f'{temporary_folder.name}\\flow_scaled.tif'
        flow_smoothed = f'{temporary_folder.name}\\flow_smoothed.tif'

        print("Preparing flow-map")
        processing.run("grass7:r.terraflow", self.param.flow_map(self.elevation_in, flow_unfiltered, 8000))
        print(f"FlowMap saved to {flow_unfiltered}")

        print("Filtering Flow-Map")
        processing.run("gdal:rastercalculator", self.param.raster_calculator(flow_unfiltered, flow_geotiff, '(A > 10)'))
        print(f"Filtered Flow-Map saved to {flow_geotiff}")

        print("Scaling Flow-Map")
        processing.run('gdal:rastercalculator',
                       self.param.raster_calculator(flow_geotiff, flow_scaled, '(A * 256)'))
        print(f"scaled Flow-map saved to {flow_scaled}")

        print("Smooting Flow-Map")
        processing.run("sagang:gaussianfilter", self.param.smoothing(flow_scaled, flow_smoothed, 10))
        print(f"Smothed Flow-Map {flow_smoothed}")
        return flow_smoothed

    def roads(self):
        slope = f'{temporary_folder.name}\\slope.tif'
        slope_smoothed = f'{temporary_folder.name}\\slope_smoothed.tif'
        slope_filtered = f'{temporary_folder.name}\\slope_filtered.tif'
        slope_clipped = f'{temporary_folder.name}\\slope_clipped.tif'
        slope_filled = f'{temporary_folder.name}\\slope_filled.tif'
        slope_cleaned = f'{temporary_folder.name}\\slope_cleaned.tif'
        slope_scaled = f'{temporary_folder.name}\\slope_scaled.tif'
        buffered_roads = f'{temporary_folder.name}\\road_buffered.shp'
        union_roads = f'{temporary_folder.name}\\road_union.shp'
        union_roads_extent = f'{temporary_folder.name}\\road_union_extent.shp'
        road_smoothed = f'{temporary_folder.name}\\road_smoothed.tif'

        print("Calculating slopes")
        processing.run("gdal:slope", self.param.slope(self.elevation_in, slope))
        print(f"Slope saved to {slope}")

        print("Smoothing slope")
        processing.run("sagang:gaussianfilter", self.param.smoothing(slope, slope_smoothed, 2))
        print(f"Slope saved to {slope_smoothed}")

        print("Filtering slope")
        processing.run("gdal:rastercalculator", self.param.raster_calculator(slope_smoothed, slope_filtered, '(A < 30)'))
        print(f"Slope filtered and saved to {slope_filtered}")

        print(f"Buffering Roads")
        processing.run("native:buffer", self.param.buffer(self.road_in, buffered_roads, 8, 500))
        print(f'Buffered Roads saved to {buffered_roads}')

        print(f'Dissolve Roads')
        processing.run("native:dissolve", self.param.dissolve(buffered_roads, union_roads))
        print(f'Roads dissolved {union_roads}')

        print(f'Clipping Union Roads')
        self.clip_vector(union_roads, union_roads_extent)
        print(f'Roads clipped {union_roads_extent}')

        print(f'Clipping Slope')
        processing.run("gdal:cliprasterbymasklayer", self.param.clip(slope_filtered, union_roads_extent, slope_clipped))
        print(f'Slope clipped {slope_clipped}')

        print("Filling empty Cells")
        processing.run("native:fillnodata", self.param.fill(slope_clipped, slope_filled))
        print(f"Filled raster written to {slope_filled}")

        print("Removing noise")
        processing.run("gdal:sieve", self.param.sieve(slope_filled, slope_cleaned, 10))
        print(f"Noise removed and written to {slope_cleaned}")

        print("Scaling values")
        processing.run('gdal:rastercalculator',
                       self.param.raster_calculator(slope_cleaned, slope_scaled, '(A * 256)'))
        print(f"Scaled written to {slope_scaled}")

        print("Smoothing raster")
        processing.run("sagang:gaussianfilter", self.param.smoothing(slope_scaled, road_smoothed, 2))
        print(f"Road-Map was saved to {road_smoothed}")

        return road_smoothed

    def masks(self, trees_in, road_in):
        trees_buffered = f'{temporary_folder.name}\\trees_buffered.shp'
        trees_buffered_clipped = f'{temporary_folder.name}\\trees_buffered_clipped.shp'
        trees_rastered = f'{temporary_folder.name}\\trees_rastered.tif'
        trees_rastered_2 = f'{temporary_folder.name}\\trees_rastered2.tif'
        trees_rastered_3 = f'{temporary_folder.name}\\trees_rastered3.tif'
        shrub_mask_temp = f'{temporary_folder.name}\\shrub1.tif'
        shrub_mask_hard = f'{temporary_folder.name}\\shrub2.tif'
        shrub_mask_smooth = f'{temporary_folder.name}\\shrub3.tif'
        road_map_inversed = f'{temporary_folder.name}\\road_inversed.tif'
        tree_hard = f'{temporary_folder.name}\\tree1.tif'
        tree_hard_scaled = f'{temporary_folder.name}\\tree2.tif'
        forest_smooth = f'{temporary_folder.name}\\tree3.tif'

        print("Buffering Trees")
        processing.run("native:buffer", self.param.buffer(trees_in, trees_buffered, 5, 5))
        print(f"Trees buffered and saved to {trees_buffered}")

        print("Clipping Trees")
        self.clip_vector(trees_buffered, trees_buffered_clipped)
        print(f"Clipped trees saved to {trees_buffered_clipped}")

        print("Rasterizing trees")
        processing.run("gdal:rasterize",
                       self.param.rasterizing(trees_buffered_clipped, trees_rastered, 256, 1009, 1009))
        print(f"Rasterized trees {trees_rastered}")

        print(f"Clipping Tree-Map {trees_rastered}")
        self.clip_raster(trees_rastered, trees_rastered_3)
        print(f"Clipped trees map {trees_rastered_3}")

        print(f"Fill No-Data-Cells")
        processing.run("native:fillnodata", self.param.fill(trees_rastered_3, trees_rastered_2))
        print(f"Filled raster written to {trees_rastered_2}")

        print("Filtering the data (Shrub)")
        processing.run("native:rasterlogicalor", self.param.filter(trees_rastered_2, road_in, shrub_mask_temp))
        print(f"Filtering the data for Shrub completed:{shrub_mask_temp}")

        print("Reversing and scaling of shrub mask")
        processing.run("gdal:rastercalculator",
                       self.param.raster_calculator(shrub_mask_temp, shrub_mask_hard, '(A * -256) + 256'))
        print(f"Reversing and scaling of shrub mask completed {shrub_mask_hard}")

        print("Smoothing")
        processing.run("sagang:gaussianfilter", self.param.smoothing(shrub_mask_hard, shrub_mask_smooth, 20))
        print(f"Shrub-Map was saved to {shrub_mask_smooth}")

        print("Inverse Roads")
        processing.run("gdal:rastercalculator", self.param.raster_calculator(road_in, road_map_inversed, 'A*-1 + 256'))
        print(f"Inversed Road saved to {road_map_inversed}")

        print("Filtering data (Forest)")
        processing.run("native:rasterbooleanand", self.param.filter(road_map_inversed, trees_rastered_2, tree_hard))
        print(f"Forest map filtered {tree_hard}")

        print("Scaling of tree mask")
        processing.run("gdal:rastercalculator", self.param.raster_calculator(tree_hard, tree_hard_scaled, '(A * 256)'))
        print(f"Scaling of tree mask completed {tree_hard_scaled}")

        print("Smoothing")
        processing.run("sagang:gaussianfilter", self.param.smoothing(tree_hard_scaled, forest_smooth, 20))
        print(f"Forest map smoothed {forest_smooth}")
        return shrub_mask_smooth, forest_smooth

    def generate_features(self):
        """
        This method extract all features in the target area, assigns Z/M (Elevation) values and saves all features to a CSV
        :return: Nothing
        """
        trees_m = self.get_m_value()
        self.save_to_csv(trees_m)

    def generate_flow(self):
        """
        This method generates a flow map based on the given elevation profile
        :return: Nothing
        """
        flow = self.flow_accumulation()
        self.save_raster(flow, self.flow_out)

    def generate_road(self):
        """
        This method generates a mask containing all the roads
        :return: The string of the temporary location of the road-Map (before it was saved to a png)
        """
        road_out = self.roads()
        self.save_raster(road_out, self.road_out)
        return road_out

    def generate_masks(self, road_in):
        """
        Generates a mask for shrub (ground cover and forest cover)
        :param road_in: The reference to the temporary road map before it was saved to a PNG
        :return:
        """
        shrub, forest = self.masks(self.trees_in, road_in)
        self.save_raster(shrub, self.shrub_out)
        self.save_raster(forest, self.forest_out)


print("#########################")
print("Tree-Visualization Script")
print("#########################")

print("This script should be run in the QGIS-Console.")
print("QGIS might freeze during the execution for several minutes.")

processor = Processor(CSV_OUT, FLOW_OUT, ROAD_OUT, FOREST_OUT, SHRUB_OUT, START_POINT_EAST, START_POINT_SOUTH, WIDTH,
                      HEIGHT, TREES_IN, ELEVATION_IN, ROADS_IN)

print("Retrieving Features")
processor.generate_features()

print("Generating Flow-Map")
processor.generate_flow()

print("Generating Road-Map")
road = processor.generate_road()

print("Generating Tree-Map")
processor.generate_masks(road)

print("Process finished")
shutil.rmtree(temporary_folder.name)
