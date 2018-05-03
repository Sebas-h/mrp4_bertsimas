class BinTreeNode:
    def __init__(self, is_leaf, parent = None):
        self.is_leaf = is_leaf
        self.id = None
        self.parent = parent
        self.child_left = None
        self.child_right = None

    def expand(self, depth):
        if depth >= 1:
            # left
            self.child_left = BinTreeNode(False, self).expand(depth - 1)
            # right
            self.child_right = BinTreeNode(False, self).expand(depth - 1)
        elif depth == 0:
            self.child_left = BinTreeNode(True, self)
            self.child_right = BinTreeNode(True, self)
        # return self to allow chaining
        return self


    def get_children(self, exclude_leafs):
        if self.child_left == None:
            return []
        elif self.child_left.is_leaf and exclude_leafs:
            return []
        return [self.child_right, self.child_left]


def generateTree(maxDepth):
	assert maxDepth > 0

	# prepare tree
	# first level 1 split (2^0), second level 2 splits (2^1), third level 4 splits (2^2) etc.
	num_branch_nodes = 0
	for x in range(0, maxDepth):
		num_branch_nodes = num_branch_nodes + 2 ** x

	num_leaf_nodes = 2 ** maxDepth

	num_nodes = num_branch_nodes + num_leaf_nodes

	print("branch-nodes: ", num_branch_nodes)
	print("leaf-nodes: ", num_leaf_nodes)

	root = BinTreeNode(False)
	root = root.expand(maxDepth - 1)
	return numberingNodes(root)


def numberingNodes(root):
    nodes = get_breadth_first_nodes(root)
    index = 1
    for node in nodes:
        node.id = index
        index = index + 1

    return root

def get_breadth_first_nodes(root, exclude_leaf_nodes = False, exclude_branch_nodes = False):
    nodes = []
    stack = [root]
    while stack:
        # set current node
        cur_node = stack[0]
        # remove current node from stack
        stack = stack[1:]
        if (not exclude_branch_nodes and not exclude_leaf_nodes) or (exclude_branch_nodes and cur_node.is_leaf) or (exclude_leaf_nodes and not cur_node.is_leaf):
            nodes.append(cur_node)
        if not cur_node.is_leaf:
            stack.append(cur_node.child_left)
            stack.append(cur_node.child_right)
    return nodes

def get_depth_first_nodes(root, exclude_leaf_nodes = False):
	nodes = []
	stack = [root]
	while stack:
		# set current node
		cur_node = stack[0]
		# remove current node from stack
		stack = stack[1:]
		nodes.append(cur_node)        
		for child in cur_node.get_children(exclude_leaf_nodes):
			stack.insert(0, child)
	return nodes
