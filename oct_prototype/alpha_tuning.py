from oct import OCT
import gurobipy

class AlphaTuning:
    
    def __init__(self, train, val_set, target, tree_depth, C_max, time_limit=300, threads=None, tol=0.0001):
        """
        #TODO: make more robust
        target in train and set must be consistent, i.e. either same column name or column number
        """
        
        
        self.tree_complexity = 0 #ignoring second term in objective function
        self.train = train
        self.val_set = val_set
        self.target = target
        self.tree_depth = tree_depth
        self.C_max = C_max
        
        self.time_limit = time_limit
        self.thread = threads
        self.tol = tol #tolerance when comparing accuracies in find_best_max_splits
        
    
    def find_best_max_splits(self):
        """
        find best C
        """
        
        results_per_c = {}
        best_c = []
        
        for C in range(1,self.C_max+1):
            o = OCT(self.train, self.target, self.tree_complexity, self.tree_depth)
        
            #add extra constraint
            o.model.addConstr(gurobipy.quicksum(self.o.d)<=C)
            o.model.update()
            o.fit(self.time_limit, self.threads)
            accuracy = o.accuracy_on_test(self.val_set, target=self.target)
            results_per_c[C] = accuracy
        
        best_obj = -1
        
        for C, obj_val in results_per_c.items():            
            if obj_val >= best_obj-self.tol:
                best_obj = obj_val
                best_c.append(C)
        
        return best_c
        
    