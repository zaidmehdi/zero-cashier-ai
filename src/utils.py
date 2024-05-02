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
        

def main():
     bbox1 = np.array([144.82, 2.0213, 1121.6, 720])
     bbox2 = np.array([187.53, 297.71, 497.89, 719.46])

     print(check_collision(bbox1, bbox2))


if __name__ == "__main__":
     main()