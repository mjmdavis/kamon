import photos
from PIL import Image, ImageDraw
from functools import partial
import matplotlib.pyplot as plt
import numpy as np


def map_to_image(imsize, coords):
    """Project a series of coordinates into an image.
    >>> map_to_image([10, 10], [[0., 0.5], [1., 0.9]])
    [(0, 5), (10, 9)]
    """
    xycoords = zip(*coords)
    pxycoords = [[int(coord * coord_max) for coord in coords]
            for coords, coord_max in zip(xycoords, imsize)]
    return list(zip(*pxycoords))


def bbox_from_points(points):
    """Create a bounding box from a list of points.
    >>> bbox_from_points([[1,2],[3,4],[5,6]])
    [1, 2, 3, 4, 5, 6]
    """
    bbox = list(points[0])
    for point in points[1:]: 
        bbox.extend(point)
    return bbox

        
def segment_image(image, areas):
    map_coords = partial(map_to_image, image.size)
    im_coords = list(map(map_coords, areas))
    im_bboxes = list(map(bbox_from_points, im_coords))
    # print(im_bboxes)
    return list(map(image.crop, im_bboxes))
    
    
def downsample(image, target_width=310):
    screen_size = sorted(image.size)
    target_size = (target_width,
                   target_width * (screen_size[1]/screen_size[0]))
    target_size = list(map(int, target_size))
    return image.resize(target_size)


def create_histogram(image):
    histvals = image.histogram()
    plt.title("Hue Histogram")
    plt.bar(range(len(histvals)), histvals)
    plt.show()
    plt.close()


def get_hue(image):
    hsvim = image.convert('HSV')
    hue_im = hsvim.split()[0]
    return hue_im


def find_colors(hist_data, tolerance=3):
    # while covered pixels < 90%
    # find max uncovered
    # count pixels within max_tol

    pixel_count = sum(hist_data)
    covered_hist = np.array(hist_data)
    peaks = list()
    color_dist = [0]
    while (sum(color_dist)/pixel_count) < 0.95:
        print(sum(color_dist)/pixel_count)
        new_peak = hist_data.index(max(covered_hist))
        peaks.append(new_peak)
        for x in range(new_peak - tolerance, new_peak + tolerance + 1):
            covered_hist[x] = 0
        color_dist = [sum([hist_data[x] for x in range(peak - tolerance, peak + tolerance + 1)])
                      for peak in peaks]
    return sorted(peaks)


def generate_sampling_coordinates(width=10, height=28):
    v_pix = 1/(height*2)
    h_pix = 1/(width*2)
    
    # top and bottom rows are a special case
    top = [(((x*2*h_pix) + (h_pix)), 0.5*v_pix) for x in range(width)]
    bottom = [(((x*2*h_pix) + (h_pix)), (height*2-0.5) * v_pix) for x in range(width)]
    # rest is uniform
    main = [((x*2*h_pix) + (h_pix), (y*2*v_pix)+ (v_pix *2)) for y in range(height-1)
            for x in range(width)]
    
    top.extend(main)
    top.extend(bottom)
    return top


def classify_hue(cclasses, hue):
        distances = [abs(cclass - hue) for cclass in cclasses] 
        found_class = distances.index(min(distances))
        return found_class


def parse_image(index=-1, picker=False):
    asset = None
    if picker:
        asset = photos.pick_asset(title='Choose Image', multi=False)
    else:
        asset = photos.get_screenshots_album().assets[index]
    # print("retrieved asset: ", asset, "\n")
    
    image = asset.get_image()
    image = downsample(image)
    # image.show()
    
    main_areas = (((0., 0.), (0.827, 1.)), 
                  ((0.827, 0.),(1., 0.603)),
                  ((0.827, 0.603), (1., 1.)))
    board, colorpicker, controls = segment_image(image, main_areas)
    
    #colorpicker.show()
    #board.show()
    
    cp_hue_im = get_hue(colorpicker)
    create_histogram(cp_hue_im)
    cp_hue_im.show()
    
    board_hue_im = get_hue(board)
    create_histogram(board_hue_im)
    board_hue_im.show()
    
    colors = find_colors(cp_hue_im.histogram())
    # print(colors)
    
    
    s_coords = generate_sampling_coordinates()
    im_coords = map_to_image(board.size, s_coords)
    
    # draw = ImageDraw.Draw(board)
    # for p in im_coords:
    #     draw.point(p, 'white')
    # board.show()
    
    # board_hue_im.show()
    hues = [board_hue_im.getpixel(point) for point in im_coords]
    
    classified = list(map(lambda x: classify_hue(colors, x), hues))
    classified = [[classified[y + (10*x)] for y in range(10)]for x in range(29)]
    # for row in classified:
    #     print(row)
    
    #return colors, classified#, [board, colorpicker, controls]
    return colors, classified, [board, colorpicker, controls]

if __name__ == "__main__":
    #colors, classified, images = parse_image(index=-4)
    colors, classified, images = parse_image(picker=True)
    print(colors)
    import matplotlib.pyplot as plt
    
    for image in reversed(images):
        image.show()
    
    #fig = plt.figure(figsize=(8,14))
    plt.imshow(classified,
        aspect='auto',
        interpolation='none')
    #plt.axis('equal')
    plt.show()
    plt.close()
