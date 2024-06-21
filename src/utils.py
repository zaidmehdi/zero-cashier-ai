import numpy as np


def check_collision(bbox1, bbox2):
        """Function to check the collision between two objects"""

        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        if (x1 < x2 + w2 and x1 + w1 > x2 and
            y1 < y2 + h2 and y1 + h1 > y2):

            return True
        else:

            return False


def is_person(n:int):
     return True if int(n) == 0 else False


def get_person_bbox(boxes:np.array, classes:np.array):
     """
     Takes arrays of detected boxes and classes and outputs the first bbox 
     labeled as 'person' and removes it from both arrays.
     """

     for i, class_id in enumerate(classes):
          if is_person(class_id):
               return boxes[i], np.delete(boxes, i, axis=0), np.delete(classes, i, axis=0)

     return None, boxes, classes


def add_to_cart(shopping_cart:set, person_bbox, boxes, classes):
    """
    Takes the person_bbox, and arrays of the other boxes and classes.
    Checks for collision between the person and every other item in boxes
    Adds items in collision to a set of items (shopping cart).
    """

    for i, bbox in enumerate(boxes):
        if not is_person(classes[i]) and check_collision(person_bbox, bbox):
            shopping_cart.add(classes[i])
    
    return shopping_cart


def remove_from_cart(shopping_cart:set, person_bbox, boxes, classes):
    """
    Removes item from cart if it is detected as not in collision with the person:
    """
     
    for i, bbox in enumerate(boxes):
        if not is_person(classes[i]) and not check_collision(person_bbox, bbox):
            shopping_cart.discard(classes[i])
    
    return shopping_cart


def get_cart_total_price(shopping_cart:set, price_map:dict):
    """Given a shopping cart, returns the total price to pay"""

    total_price = 0
    for item in shopping_cart:
         total_price += price_map[item]
    
    return total_price


def main():
     bbox1 = np.array([144.82, 2.0213, 1121.6, 720])
     bbox2 = np.array([187.53, 297.71, 497.89, 719.46])

     print(check_collision(bbox1, bbox2))


if __name__ == "__main__":
     main()