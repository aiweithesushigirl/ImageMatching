        #Get data with label from 0 and 9
        idx_list = []
        idx_0 = np.where(mnist_train_labels[:] == 0)[0]
        images_0 = train_images[idx_0]
        idx_1 = np.where(mnist_train_labels[:] == 1)[0]
        images_1 = train_images[idx_1]
        idx_2 = np.where(mnist_train_labels[:] == 2)[0]
        images_2 = train_images[idx_2]
        idx_3 = np.where(mnist_train_labels[:] == 3)[0]
        images_3 = train_images[idx_3]
        idx_4 = np.where(mnist_train_labels[:] == 4)[0]
        images_4 = train_images[idx_4]
        idx_5 = np.where(mnist_train_labels[:] == 5)[0]
        images_5 = train_images[idx_5]
        idx_6 = np.where(mnist_train_labels[:] == 6)[0]
        images_6 = train_images[idx_6]
        idx_7 = np.where(mnist_train_labels[:] == 7)[0]
        images_7 = train_images[idx_7]
        idx_8 = np.where(mnist_train_labels[:] == 8)[0]
        images_8 = train_images[idx_8]
        idx_9 = np.where(mnist_train_labels[:] == 9)[0]
        images_9 = train_images[idx_9]

        idx1_list = []
        idx1_0 = np.where(mnist_test_labels[:] == 0)[0]

        images1_0 = test_images[idx1_0]
        idx1_1 = np.where(mnist_test_labels[:] == 1)[0]
        images1_1 = test_images[idx1_1]
        idx1_2 = np.where(mnist_test_labels[:] == 2)[0]
        images1_2 = test_images[idx1_2]
        idx1_3 = np.where(mnist_test_labels[:] == 3)[0]
        images1_3 = test_images[idx1_3]
        idx1_4 = np.where(mnist_test_labels[:] == 4)[0]
        images1_4 = test_images[idx1_4]
        idx1_5 = np.where(mnist_test_labels[:] == 5)[0]
        images1_5 = test_images[idx1_5]
        idx1_6 = np.where(mnist_test_labels[:] == 6)[0]
        images1_6 = test_images[idx1_6]
        idx1_7 = np.where(mnist_test_labels[:] == 7)[0]
        images1_7 = test_images[idx1_7]
        idx1_8 = np.where(mnist_test_labels[:] == 8)[0]
        images1_8 = test_images[idx1_8]
        idx1_9 = np.where(mnist_test_labels[:] == 9)[0]
        images1_9 = test_images[idx1_9]
        print("single shape", images_9.shape)
        # images_tgt = np.concatenate(
        #     (images_0[:500, :], images_1[:500, :], images_2[:500, :],
        #      images_3[:500, :], images_4[:500, :], images_5[:500, :],
        #      images_6[:500, :], images_7[:500, :], images_8[:500, :],
        #      images_9[:500, :]),
        #     axis=0)
        images_tgt = np.concatenate(
            (images_0[:500], images_1[:500], images_2[:500], images_3[:500],
             images_4[:500], images_5[:500], images_6[:500], images_7[:500],
             images_8[:500], images_9[:500]),
            axis=0)
        images1_tgt = np.concatenate(
            (images1_0[:10], images1_1[:10], images1_2[:10], images1_3[:10],
             images1_4[:10], images1_5[:10], images1_6[:10], images1_7[:10],
             images1_8[:10], images1_9[:10]),
            axis=0)

        print("imagetgt shape", images_tgt.shape)

        def visualize_data(principal_components):
        fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)

    ax.set_title('2 component PCA', fontsize=20)

    targets = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']
    colors = ['r', 'b', 'g', 'c', 'm', 'y', 'k', 'C7', 'C8', 'C9', 'C5']

    ax.scatter(
        principal_components[:, 0],
        principal_components[:, 1],
        c=colors[0],
        s=50)

    ax.scatter(
        principal_components[500:999, 0],
        principal_components[500:999, 1],
        c=colors[1],
        s=50)

    ax.scatter(
        principal_components[1000:1499, 0],
        principal_components[1000:1499, 1],
        c=colors[2],
        s=50)

    ax.scatter(
        principal_components[1500:1999, 0],
        principal_components[1500:1999, 1],
        c=colors[3],
        s=50)

    ax.scatter(
        principal_components[2000:2499, 0],
        principal_components[2000:2499, 1],
        c=colors[4],
        s=50)

    ax.scatter(
        principal_components[2500:2999, 0],
        principal_components[2500:2999, 1],
        c=colors[5],
        s=50)

    ax.scatter(
        principal_components[3000:3499, 0],
        principal_components[3000:3499, 1],
        c=colors[6],
        s=50)

    ax.scatter(
        principal_components[3500:3999, 0],
        principal_components[3500:3999, 1],
        c=colors[7],
        s=50)

    ax.scatter(
        principal_components[4000:4499, 0],
        principal_components[4000:4499, 1],
        c=colors[8],
        s=50)

    ax.scatter(
        principal_components[4500:4999, 0],
        principal_components[4500:4999, 1],
        c=colors[9],
        s=50)

    ax.legend(targets)
    fig.savefig("grid.png")


    def get_principal_components(image_01):
        pca = PCA(n_components=2)
    X = image_01
    principal_components = pca.fit_transform(X)
    # print(principal_components.shape)

    #display figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('PC1', fontsize=15)
    ax.set_ylabel('PC2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)

    targets = ['1', '0']
    colors = ['r', 'b']
    ax.scatter(
        principal_components[0:499, 0],
        principal_components[0:499, 1],
        c=colors[0],
        s=50)
    ax.scatter(
        principal_components[500:999, 0],
        principal_components[500:999, 1],
        c=colors[1],
        s=50)
    ax.legend(targets)
    fig.savefig("grid.png")


        # principal_components_c = (x_std.dot(ten_eigenvec))
    # print("x_std", x_std.shape, "matrix_w", ten_eigenvec.shape)
    # pca = PCA(n_components=2)
    # print("combine shape", combined_img.shape, test_instance.shape)
    # X = combined_img
    # gray_test = cv2.cvtColor(test_instance, cv2.COLOR_BGR2GRAY)

    # Y = gray_test

    # principal_components = pca.fit_transform(X)
    # principal_components_test = pca.transform(Y)

    # print("transformed pca", principal_components.shape,
    #       principal_components_test)

    # neighbors = getNeighbors(principal_components, principal_components_test,
    #                          11)

    def check_number(location):
        if location >= 0 and location < 500:
        return 0
    elif location >= 500 and location < 1000:
        return 1
    elif location >= 1000 and location < 1500:
        return 2
    elif location >= 1500 and location < 2000:
        return 3
    elif location >= 2000 and location < 2500:
        return 4
    elif location >= 2500 and location < 3000:
        return 5
    elif location >= 3000 and location < 3500:
        return 6
    elif location >= 3500 and location < 4000:
        return 7
    elif location >= 4000 and location < 4500:
        return 8
    elif location >= 4500 and location <= 5000:
        return 9