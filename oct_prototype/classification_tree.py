class BinClassNode:
    
    def __init__(self, is_leaf, applies_split=None, split_on=None, split_value=None, label=None, right_children=None, left_children=None):
        
        self.is_leaf = is_leaf #boolean
        self.applies_split = applies_split #boolean
        self.split_on = split_on # feature (number) to split on, integer
        self.split_value = split_value # split threshold
        self.label = label # if leaf node: integer indicating the leaf nodes class number
        #self.right_children = right_children # node (odd number)
        #self.left_children = left_children # node (even number)
        
    def __str__(self):
        if self.is_leaf:
            return 'Leaf Node: {1} \nlabel: {0}\n*****'.format(self.label, self.is_leaf)        
        else:
            return 'Branch node\napplies a split: {0}\nsplits on: {1}\nsplit value: {2}\n*****'.format(self.applies_split, self.split_on, self.split_value)


class BinClassificationTree:
    
    def __init__(self, n_total_nodes, model, n_total_classes, n_features):
        """
        #TODO: current implementation relies on variable naming, should be more robust to changes in variable names
        model = (optimized) gurobi model
        """
        
        self.n_total_nodes = n_total_nodes
        self.n_total_classes = n_total_classes
        self.n_features = n_features
        self.branch_threshold = int((self.n_total_nodes+1)/2)
        self.tree = {}
        self.model = model
        self.create_tree()        
        
    def __str__(self):
        #TODO: ascii art
        s = ''
        for node in self.tree.keys():
            s+=node+'\n'
            s+= str(self.tree.get(node).get('node_data'))+'\n'
        return s
    
        
    def create_tree(self):
        
        for i in range(1, self.n_total_nodes+1):
            is_leaf = i>=self.branch_threshold
            applies_split, split_on, split_value, label = self.get_solution_vars(is_leaf, node_no=i)
            node = BinClassNode(is_leaf=is_leaf,
                                applies_split=applies_split,
                                split_on=split_on,
                                split_value=split_value,
                                label=label)
            
            self.tree['node_'+str(i)] = {'left_child':'node_'+str(i*2),
                                          'right_child':'node_'+str((i*2)+1),
                                          'node_data': node}
            
        
    def get_solution_vars(self, is_leaf, node_no):
        
        applies_split = None
        split_on = None
        split_value = None
        label = None
        
        if is_leaf:
            # find label assigned to leaf
            #TODO: node does not contain any point
            for l in range(self.n_total_classes):
                var_name = 'label_{0}_is_assigned_to_node_{1}'.format(l, node_no)
                #var_name = 'label_0_is_assigned_to_node_2'
                v = self.model.getVarByName(var_name).X
                if v == 1:
                    label = l
                    break
        else:
            var_name = 'node_{0}_applies_split'.format(node_no)
            v = self.model.getVarByName(var_name).X
            if v == 1:
                applies_split = True
                #find which feature the node splits on
                for f in range(self.n_features):
                    var_name = 'node_{0}_splits_on_feature_{1}'.format(node_no, f)
                    value = self.model.getVarByName(var_name).X
                    if value==1:
                        split_on = f
                #find split value (=b) of that node
                var_name = 'split_value_node_{0}'.format(node_no)
                split_value = self.model.getVarByName(var_name).X
            else:
                applies_split = False
                
        return applies_split, split_on, split_value, label
    
    def predict(self, df, cols):
        root = 'node_1'
        predictions = []
        #TODO: efficiency
        for row_no in range(len(df)):
            x = df[cols].iloc[row_no].values
            pred = self._predict(x, root)
            #print('Prediction for {0}: {1}'.format(x, pred))
            predictions.append(pred)
            
        return predictions
            
    def _predict(self, x, node):
        """
        node: string indicating node number, e.g. node_1 (corresponds to naming in tree)
        """
        n = self.tree.get(node).get('node_data')
        if n.is_leaf:
               
            r_label = n.label
            #print('{0} is leaf. Label: {1}'.format(node, r_label))
            return r_label
    
        else:
            #apply split until leaf node is reached
            if n.applies_split:
                a = n.split_on
                b = n.split_value
                if x[a]<b:
                    return self._predict(x, self.tree.get(node).get('left_child'))
                else:
                    return self._predict(x, self.tree.get(node).get('right_child'))
            else:
                return self._predict(x, self.tree.get(node).get('right_child'))
            
        