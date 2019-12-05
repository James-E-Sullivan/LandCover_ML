import affine


def get_corners(raster):
    """
    Obtains corner coordinates from a raster image
    :param raster: Input raster image
    :return corners: Four corner coordinates, as a tuple of 4 tuples
    """
    transform = raster.GetGeoTransform()

    # transforms
    x_origin = transform[0]      # min x
    pixel_width = transform[1]
    rot1 = transform[2]
    y_origin = transform[3]      # max y
    rot2 = transform[4]
    pixel_height = transform[5]

    # x,y min,max
    min_x = x_origin
    max_y = y_origin
    max_x = x_origin + pixel_width * raster.RasterXSize
    min_y = y_origin + pixel_height * raster.RasterYSize

    # test to see if this is top corners
    top_left = (min_x, max_y)
    top_right = (max_x, max_y)
    bot_left = (min_x, min_y)
    bot_right = (max_x, min_y)

    corners = (top_left, top_right, bot_left, bot_right)
    return corners


def get_pixel_value(geo_coord, data_source):
    """
    Returns a floating point value that corresponds to a given point.
    :param geo_coord: Coordinates of wanted pixel value in a raster
    :param data_source: Parent raster, in which the coordinates should reside
    :return pixel_coord: floating point pixel coordinates
    """

    # unpack x and y geo coordinates
    x, y = geo_coord[0], geo_coord[1]

    # define reverse transform of geo-coordinates
    forward_transform = affine.Affine.from_gdal(*data_source.GetGeoTransform())
    reverse_transform = ~forward_transform

    # obtain pixel values corresponding to geo-coordinates
    px, py = reverse_transform * (x, y)
    px, py = int(px + 0.5), int(py + 0.5)  # center-point of pixels
    pixel_coord = px, py

    return pixel_coord

