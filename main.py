import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils import findAverageColor, findDominantColors, plotColors, find_circle, find_circle2, distance


def quantify_colors(imgpath, numberofcolors):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # pixels = np.float32(img.reshape(-1, 3))

    dominant, palette, counts = findDominantColors(img, numberofcolors)
    plotColors(img, palette, counts)


    return dominant, palette, counts, img



if __name__ == '__main__':
    imgpath = 'image.tif'
    numberofcolors = 5
    # dominant, palette, counts, img = quantify_colors(imgpath, numberofcolors)
    # print(dominant, palette, counts)
    img = cv2.imread(imgpath)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    edges = find_circle2(img)

    contours = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    big_contour = max(contours, key=cv2.contourArea)
    # print(contours)
    largest_contour_index, largest_contour = max(enumerate(contours), key=lambda x: cv2.contourArea(x[1]))
    # draw white filled contour on black background
    result = np.zeros_like(img)
    cv2.drawContours(result, [big_contour], 0, (255,255,255), cv2.FILLED)

    # print(big_contour)
    (x_center, y_center), radius = cv2.minEnclosingCircle(big_contour)
    i = contours[largest_contour_index]
    M = cv2.moments(i)
    if M['m00'] != 0:
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        cv2.drawContours(img, [i], -1, (0, 255, 0), 2)
        cv2.circle(img, (cx, cy), 7, (255, 0, 0), -1)
        cv2.putText(img, "center", (cx - 20, cy - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
    # print(f"x: {cx} y: {cy}")
    # cv2.imshow('img', img)
    # cv2.waitKey(0)

    center = (cx, cy)
    numberofcircles = 20
    pixel_per_area = 25
    all_circles = []
    mean_colors = []
    for each in range(numberofcircles):
        # print(each)
        mask = np.zeros_like(img)
        circle = cv2.circle(img, center, int(radius+pixel_per_area*(each+1)), (255, each*10, 0), 2)
        cv2.circle(mask, center, int(radius + pixel_per_area * (each + 1)), (255, 255, 255), thickness=-1)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        masked_img = cv2.bitwise_and(img, img, mask=mask_gray)
        pixels = masked_img.reshape(-1, 3)
        non_black_pixels_mask = np.all(pixels != [0, 0, 0], axis=1)
        non_black_pixels = pixels[non_black_pixels_mask]
        mean_color = np.mean(non_black_pixels, axis=0) if non_black_pixels.size else [0, 0, 0]
        # print(f"Average color for circle {each}: {mean_color}")
        mean_colors.append(mean_color)

    # Optional: Draw the circle on the image for visualization
        cv2.circle(img, center, int(radius + pixel_per_area * (each + 1)), (0, 255, 0), 2)
        all_circles.append(circle)
    
    # cv2.imshow('Circles with Average Colors', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # Create an empty mask for the current circle
    
    result_image = np.zeros_like(img)


# Initialize the previous radius to start drawing from the center
    previous_radius = 0

    for each in range(numberofcircles):
        # Calculate the current circle's radius
        current_radius = int(radius + pixel_per_area * (each + 1))
        
        # Create an empty mask for the current circle
        mask = np.zeros_like(img)
        cv2.circle(mask, center, current_radius, (255, 255, 255), thickness=-1)
        
        # Convert mask to grayscale (if your image is color)
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
        # Find the pixels in the original image that fall within the current circle but outside the previous circle
        masked_img = cv2.bitwise_and(img, img, mask=mask_gray)

        # Calculate the average color of these pixels
        pixels = masked_img.reshape(-1, 3)
        non_black_pixels_mask = np.all(pixels != [0, 0, 0], axis=1)
        non_black_pixels = pixels[non_black_pixels_mask]
        
        mean_color = np.mean(non_black_pixels, axis=0) if non_black_pixels.size else [0, 0, 0]

        # Create a mask for the area between the current and previous circle
        circle_mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.circle(circle_mask, center, current_radius, 255, thickness=-1)
        cv2.circle(circle_mask, center, previous_radius, 0, thickness=-1)  # Remove inner circle

        # Fill the area between the current and previous circle with the average color on the result image
        result_image[circle_mask == 255] = mean_color

        # Update the previous radius
        previous_radius = current_radius


    # Show the final image with concentric circles filled with their average colors
    cv2.imshow('Filled Circles', result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


    d = distance(mean_colors[0], mean_colors[1])


    print(mean_colors[0])

    distances = []
    for index, each in enumerate(mean_colors[:-1]):
        d = distance(mean_colors[index], mean_colors[index+1])
        distances.append(d)

    plt.plot(distances)
    plt.show()
    plt.imshow(result_image)
    plt.show()
    print(distances)



    # Convert mask to grayscale (if your image is color)
   
    
    # Find the pixels in the original image that fall within the current circle
    
    
    # Calculate the average color of these pixels
    # Convert the masked image to a list of pixels and remove black background pixels
    
    
    
    
    # Calculate the average color
    
    
    

# Show the final image with drawn circles (if not running in headless mode)


    # print(type(region_of_interest))


    

   


    # grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # edges = np.uint(0.5*edges)
    # grey = np.uint(0.5*grey)
    # combined_img = edges + grey
    # cv2.imshow('combined', combined_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # find_circle(edges)
