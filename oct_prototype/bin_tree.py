class BinTreeNode:
    def __init__(self, is_leaf, parent=None):
        self.is_leaf = is_leaf
        self.id = None
        self.parent = parent
        self.child_left = None
        self.child_right = None
        self.cart_id = None
        self.cart_leaf = False
        self.applies_split = False
        self.threshold = 0
        self.split_feature = None
        self.value = None
        self.label = None
        self.datapoints = []
        self.num_datapoints = 0
        self.num_missclass = 0


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

    def get_children(self, exclude_leafs=False):
        if self.child_left is None:
            return []
        elif self.child_left.is_leaf and exclude_leafs:
            return []
        return [self.child_right, self.child_left]

    def get_left_child(self, exclude_leafs=False):
        if not self.get_children(exclude_leafs):
            return None
        else:
            return self.get_children(exclude_leafs)[1]

    def get_right_child(self, exclude_leafs=False):
        if not self.get_children(exclude_leafs):
            return None
        else:
            return self.get_children(exclude_leafs)[0]

    def get_ancestors(self):
        ancestors = []
        ancestors_left = []
        ancestors_right = []
        current_node = self
        while current_node.parent is not None:
            ancestors.append(current_node.parent)
            if current_node.parent.child_left == current_node:
                ancestors_left.append(current_node.parent)
            else:
                ancestors_right.append(current_node.parent)
            current_node = current_node.parent
        return ancestors, ancestors_left, ancestors_right

    def get_parent(self):
        return self.parent

    def get_node_by_id(self, id):
        nodes = get_breadth_first_nodes(self)

        for node in nodes:
            if node.id == id:
                return node

        return None

    def get_node_by_cart_id(self, id):
        nodes = get_breadth_first_nodes(self)

        for node in nodes:
            if node.cart_id == id:
                return node

        return None


def generateTree(maxDepth):
    assert maxDepth > 0

    # num_nodes = (2 ** (maxDepth + 1)) - 1
    # # number of leaf nodes
    # num_leaf_nodes = 2 ** maxDepth
    # # number of branch nodes
    # num_branch_nodes = num_nodes - num_leaf_nodes

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


def get_breadth_first_nodes(root, exclude_leaf_nodes=False, exclude_branch_nodes=False):
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


def get_depth_first_nodes(root, exclude_leaf_nodes=False):
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
