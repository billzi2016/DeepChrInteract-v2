# -*- coding:utf-8 -*-

import argparse

if __name__ == '__main__':

    # Basic description
    print('*'*36)
    print('*'*36)
    print()
    print('    Welcome to use DeepChrInteract')
    print()
    print('*'*36)
    print('*'*36)
    print()

    ####################
    # Parser
    ####################
    parser = argparse.ArgumentParser()

    # Introduce a time stamp
    import time
    print( 'begin time >>> ' + time.asctime(time.localtime(time.time())) + '\n\n\n' )

    # Statistics Time
    start = time.time()

    # Preprocessing
    print('If it is the first time to use, please preprocess the data first.')
    print('Use the command: python3 DeepChrInteract.py -p true -n [gene name]')
    print('For example: python3 DeepChrInteract.py -p true -n AD2.po')
    print()

    ####################
    # Parameters: preprocessing
    ####################
    parser.add_argument('-p', '--preprocessing',
                        help = 'Preprocess the data, if you enter [true] (case sensitive), then proceed, if no, pass this process. Note: This command only needs to be entered once.',
                        required = False)

    ####################
    # Parameters: select model
    ####################
    parser.add_argument('-m', '--model',
                        help = 'Enter the model name which your choose: [onehot_cnn_one_branch] / [onehot_cnn_two_branch] / [onehot_embedding_dense] / [onehot_embedding_cnn_one_branch] / [onehot_embedding_cnn_two_branch] / [onehot_dense] / [onehot_resnet18] / [embedding_cnn_one_branch] / [embedding_cnn_two_branch] (all use lowercase).',
                        required = False)

    ####################
    # Parameters: choose the type, training or testing
    ####################
    parser.add_argument('-t', '--type',
                        help = 'Please choose [train] / [test] (all use lowercase).',
                        required = False)

    ####################
    # Parameters: selected gene name
    ####################
    parser.add_argument('-n', '--name',
                        help = 'Enter the gene name of your choice (note: case sensitive). Here is the source gene name.',
                        required = False)

    ####################
    # Parameters: verified genes
    ####################
    parser.add_argument('-o', '--object',
                        help = 'Enter the gene name of your choice (note: case sensitive). Here is the object gene name.',
                        required = False)

    ####################
    # Parameters: gene length
    ####################
    parser.add_argument('-l', '--length',
                        default = 10001,
                        help = 'Enter the length of gene, default is 10001.',
                        required = False)

    args = parser.parse_args()

    print('\n=== Below is your input ===\n')
    if args.preprocessing == 'true':
        print('args.preprocessing = '+ args.preprocessing)
        print('args.name = '+ args.name)
    else:
        print('args.model = '+ args.model)
        print('args.type = '+ args.type)
        print('args.name = '+ args.name)
        print('args.length = '+ str(args.length))
        try:
            print('args.object = '+ args.object)
        except:
            pass
    print('===========================')
    print()

    # Data preprocessing
    if args.preprocessing == 'true':
        
        from data_preprocessing import data_prep
        data_prep(args.name)
        
    else:

        from train import train_cnn_dense_resnet, train_cnn_separate, train_embedding
        from test import test_cnn_dense_resnet, test_cnn_separate, test_embedding


        ####################
        # Input error prompt
        ####################
        if args.model not in ['onehot_cnn_one_branch', 'onehot_cnn_two_branch', 'onehot_embedding_dense', 
                              'onehot_dense', 'onehot_resnet18',
                              'embedding_cnn_one_branch', 'embedding_cnn_two_branch',
                              'onehot_embedding_cnn_one_branch', 'onehot_embedding_cnn_two_branch']:
            print("Wrong model name!\nUse command python3 DeepChrInteract.py -h to see the correct model name!")
            exit(1)
            
        if args.type not in ['train', 'test']:
            print("Wrong type name!\nType must in ['train', 'test']")
            exit(1)
            
        print() # Print a blank line


        ########################################
        #
        # File path system
        #
        ########################################
        import os
         
        def mkdir(path):
            if not os.path.exists(path): # Determine whether there is a folder, if it does not exist, create a folder
                os.makedirs(path) # makedirs This path will be created if the path does not exist when creating a file
                print('-> make new folder: ' + path)
            else:
                print('-> ' + path + ' folder already exist. pass.')
                        
        # Use mkdir (relative path/absolute path can be used) # call function


        # Create path
        mkdir('h5_weights/' + args.name) # Just need the gene name

        mkdir('result/' + args.name + '/onehot_cnn_one_branch')
        mkdir('result/' + args.name + '/onehot_cnn_two_branch')
        mkdir('result/' + args.name + '/onehot_embedding_dense')

        mkdir('result/' + args.name + '/onehot_dense')
        mkdir('result/' + args.name + '/onehot_resnet18')
        mkdir('result/' + args.name + '/onehot_resnet34')

            
        mkdir('result/' + args.name + '/embedding_cnn_one_branch')
        mkdir('result/' + args.name + '/embedding_cnn_two_branch')

        mkdir('result/' + args.name + '/onehot_embedding_cnn_one_branch')
        mkdir('result/' + args.name + '/onehot_embedding_cnn_two_branch')


        if args.model=='onehot_cnn_one_branch' or args.model=='onehot_embedding_dense' \
           or args.model=='onehot_dense' or args.model=='onehot_resnet18':
   
            if args.type=='train':
                train_cnn_dense_resnet(args.name, args.model, int(args.length))
            elif args.type=='test':
                test_cnn_dense_resnet(args.name, args.model, args.object, int(args.length))


        if args.model == 'onehot_cnn_two_branch':
            
            if args.type=='train':
                train_cnn_separate(args.name, args.model, int(args.length))
            elif args.type=='test':
                test_cnn_separate(args.name, args.model, args.object, int(args.length))


        if args.model=='embedding_cnn_one_branch' or args.model=='embedding_cnn_two_branch' \
           or args.model=='onehot_embedding_cnn_one_branch' or args.model=='onehot_embedding_cnn_two_branch':

            if args.type=='train':
                train_embedding(args.name, args.model)
            elif args.type=='test':
                test_embedding(args.name, args.model, args.object)


    # Introduce a timestamp
    import time
    print( 'end time >>> ' + time.asctime(time.localtime(time.time())) + '\n\n\n\n\n\n\n\n\n\n' )

    end = time.time()
    print()
    print()
    try:
        print('args.model = '+ args.model)
    except:
        pass
    print('time used = ' + str(end-start))
    print()
    print()


    ####################
    # Avoid crash, test use
    ####################
    # import os
    # os.system('pause')
