from pm4py.objects.process_tree.obj import Operator
from pm4py.objects.process_tree.utils import generic
from copy import deepcopy
import itertools
import time

def get_layer_info(tree_node, percentage):
    root = tree_node._get_root()
    xor_n = 0
    xor_sum = 0
    layer_xor = list()
    if root.operator == Operator.XOR:
        xor_n += 1
        xor_sum += 1
    layer_xor.append(xor_n)

    current_nodes = root.children

    change_of_leaves = True
    while change_of_leaves:
        xor_n = 0
        new_nodes = list()
        for child in current_nodes:
            if child.operator is not None:
                if child.operator == Operator.XOR:
                    xor_n += 1
                    xor_sum += 1
                if child.operator != Operator.LOOP:
                    for node in child.children:
                        new_nodes.append(node)

        layer_xor.append(xor_n)

        if len(new_nodes) >= 1:
            current_nodes = new_nodes
        else:
            change_of_leaves = False
    # print(layer_xor)
    n = xor_sum // percentage
    # print("the number of xor to decompose:", n)
    c_xor_n = 0
    if len(layer_xor) > 1:
        for i in range(len(layer_xor)):
            c_xor_n += layer_xor[i]
            if c_xor_n >= n:
                cut_num = n-(c_xor_n-layer_xor[i])
                return i, n, cut_num


def get_leaves_n(current_node, cut_n):
    t = 0
    if len(current_node.children) > 0:
        c_list = deepcopy(current_node.children)
    while len(c_list) > 0:
        c = c_list.pop()
        if c.label is not None:
            t = t+1
            # print(t)
            if t >= cut_n:
                return t
        else:
            for i in deepcopy(c.children):
                c_list.append(i)

    return t


# compute the cost of model
def compute_cost(node, cost):
    if node.operator is None:
        if node.label is not None:
            return 10000
        else:
            return 1
    else:
        if node.operator == Operator.SEQUENCE:
            for child in node.children:
                if child.operator is None:
                    cost += compute_cost(child, cost)
                else:
                    cost = compute_cost(child, cost)
                # print(child, ",", cost)

        if node.operator == Operator.XOR:
            select = list()

            for child in node.children:
                if child.operator is None:
                    cost += compute_cost(child, cost)
                    return cost
                else:
                    select.append(child)
            if len(select) == len(node.children):
                m = compute_cost(select[0], cost)
                # print("m", m)
                for opt in select[1:]:
                    n = compute_cost(opt, cost)
                    # print("n", n)
                    if n < m:
                        m = n
                cost = m
                # print(cost)

            # print(child, "xor", cost)
        if node.operator == Operator.PARALLEL:
            for child in node.children:
                if child.operator is None:
                    cost += compute_cost(child, cost)
                else:
                    cost = compute_cost(child, cost)
                # print(child, "*", cost)
        if node.operator == Operator.LOOP:
            child = node.children[0]
            if child.operator is None:
                cost += compute_cost(child, cost)
            else:
                cost = compute_cost(child, cost)

    return cost


def get_tree_cost(tree):
    root = tree._get_root()
    current_node = root
    tree_cost = compute_cost(current_node, 0)
    return tree_cost

def delete_child(current_node, tree):

    while current_node.children:
        c = current_node.children[0]
        p = c.parent

        p.children.remove(c)


def set_current_child(tr, index, node):
    node.children[index] = tr
    tr._set_parent(node)

def merge(node,list_trees):
    for comb_trees in list_trees:
        for tr, index in comb_trees.items():
            set_current_child(tr, index[1], node)


def decomposetree(current_tree, cut_layer, current_node, layer_index, xor_cut,last_nodes):
    global xor_n_s
    global last_nodes_n
    # print(last_nodes_n)
    if layer_index[0] > cut_layer:
        c_list = list()
        c_r = {current_tree: layer_index}
        c_list.append(c_r)
        return c_list
    if layer_index[0] == cut_layer:
        if current_node.operator == Operator.XOR:
            last_nodes_n += 1
            # print("last_nodes",last_nodes_n)
            if last_nodes_n > last_nodes:
                c_list = list()
                c_r = {current_tree: layer_index}
                c_list.append(c_r)
                return c_list

    print("xor_n",xor_n_s)
    # print("current_node",current_node.operator)
    # if current_node.operator ==Operator.XOR:
    #     print(current_node)
    if xor_n_s >= xor_cut:
        # print("xor_n: ",xor_n_s)
        c_list = list()
        c_r = {current_tree: layer_index}
        c_list.append(c_r)
        return c_list



    if current_node.operator == Operator.SEQUENCE or current_node.operator == Operator.PARALLEL:
        children_list = current_node.children
        flag = 0
        current_tree_dic = dict()

        for child_num in range(len(children_list)):
            if children_list[child_num].operator == Operator.SEQUENCE or children_list[child_num].operator == Operator.PARALLEL or children_list[child_num].operator == Operator.XOR:
                # if children_list[child_num].operator == Operator.XOR:
                #     xor_n += 1

                l_i = list()
                l_i.append(layer_index[0]+1)
                l_i.append(child_num)
                current_tree_dic[children_list[child_num]] = l_i
                flag = 1
        if flag == 0:
            c_list = list()
            c_r = {current_tree: layer_index}
            c_list.append(c_r)
            return c_list
        else:
            result_tree = list()
            save_trees = list()
            for child_tree, la_in in current_tree_dic.items():
                
                d_tree = deepcopy(child_tree)
                d_tree.parent = None
                root = d_tree._get_root()
                # if root.operator ==Operator.XOR:
                #     xor_n_s += 1
                trees = decomposetree(d_tree, cut_layer, root, la_in, xor_cut,last_nodes)
                save_trees.append(trees)
            # print(save_trees)

            z = list(itertools.product(*save_trees))

            # print(z)
            for zz in z:
                merge(current_node, zz)
                g_tree = deepcopy(current_tree)
                result_tree.append({g_tree: layer_index})

            return result_tree

    elif current_node.operator == Operator.XOR:
        xor_n_s += 1
        children_list = current_node.children
        result_trees = list()
        for child_num in range(len(children_list)):
            child = children_list[child_num]
            if child.operator == Operator.SEQUENCE or child.operator == Operator.PARALLEL or child.operator == Operator.XOR:
                
                d_tree = deepcopy(child)
                d_tree.parent = None
                root = d_tree._get_root()
                # if root.operator ==Operator.XOR:
                #     xor_n_s += 1
                trees = decomposetree(d_tree, cut_layer, root, [layer_index[0]+1, layer_index[1]], xor_cut,last_nodes)
                for current_result in trees:
                    result_trees.append(current_result)
            else:
                d_tree = deepcopy(child)
                d_tree.parent = None
                current_result = {d_tree: layer_index}
                result_trees.append(current_result)
        return result_trees


xor_n_s = 0
last_nodes_n=0
def apply(tree, perc):
    root = tree._get_root()

    current_node = root
    layer_index = [0, 0]
    print("decompose is starting!")
    cut_layer,xor_cut, last_nodes=get_layer_info(tree, perc)
    print("cut_layer", cut_layer)
    print("the number of xor to decompose",xor_cut)

    sub_trees_list = decomposetree(tree, cut_layer, current_node, layer_index, xor_cut,last_nodes)

    print("decompose is finished!")
    print("the number of sub_trees: ",len(sub_trees_list))
    r = list()
    for ite in sub_trees_list:
        for s_tree, nnn in ite.items():
            r.append(s_tree)

    return r

