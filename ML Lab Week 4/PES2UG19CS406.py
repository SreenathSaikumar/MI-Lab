import numpy as np


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):

        self.weighted = weighted
        self.k_neigh = k_neigh
        self.p = p

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """

        self.data = data
        self.target = target.astype(np.int64)

        return self

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        rowdata,coldata=self.data.shape
        rowx,colx=x.shape
        distmat=np.zeros((rowx,rowdata))
        for i in range(rowx):
            sel=x[i]
            for k in range(rowdata):
                sel2=self.data[k]
                dist=0
                for j in range(colx):
                    dist+=np.absolute((sel2[j]-sel[j])**self.p)
                dist=(dist)**(1/self.p)
                distmat[i][k]=dist
        self.distmat=distmat
        return distmat

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        distmat=self.find_distance(x)
        row,col=distmat.shape
        k=self.k_neigh
        neigh_dists=[]
        idx_of_neigh=[]
        for i in range(row):
            sel=distmat[i]
            sel2=np.sort(sel)
            idx_slice=np.argsort(sel)
            neigh_dists.append(sel2[:k])
            idx_of_neigh.append(idx_slice[:k])
        return [neigh_dists,idx_of_neigh]

    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        indexvals=np.array(self.k_neighbours(x)[1])
        dist=np.array(self.k_neighbours(x)[0])
        res=[]
        if self.weighted==False:
            for i in indexvals:
                temp=[]
                for j in i:
                    temp.append(self.target[j])
                try:
                    res.append(temp[np.argmax(np.unique(temp,return_counts=True)[1])])
                except:
                    res.append(temp[0])
        else:
            rowval=0
            for i in indexvals:
                dicts={}
                col=0
                for j in i:
                    try:
                        dicts[self.target[j]]=dicts[self.target[j]]+(1/dist[rowval][col])
                    except:
                        dicts[self.target[j]]=1/dist[rowval][col]
                    col+=1
                maxval=max(dicts,key= lambda x: dicts[x])
                res.append(maxval)
            rowval+=1
        return res

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        # TODO
        ctr=0
        res=self.predict(x)
        for i in range(len(res)):
            if res[i]==y[i]:
                ctr+=1
        return (ctr/len(res)*100)
