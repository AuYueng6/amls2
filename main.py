import test

test_set=False #the test process needs test set
demo=True
scale=4 #set the factor for demo function

if __name__=='__main__':
    network=['SRCNN','FSRCNN']
    if test_set:
        for i in network:
            test.test_model(i) #test the models as described in the report

    if demo:
        for i in network:
            test.demo(i,scale) #run the demo function to generate SR images
