from blackboard import BlackBoard, Prediction
import cv2


if __name__=="__main__":
    # Create a black image, a window and bind the function to window
    predictions=Prediction("MNIST_model.hdf5")
    blackboard=BlackBoard(predictions)
    blackboard.blackboard_size(500)
    blackboard.line_size(10)
    
    while(1):
        blackboard.show()
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        elif k == ord('c'):
            blackboard.erase()

    cv2.destroyAllWindows()


