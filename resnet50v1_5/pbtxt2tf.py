from __future__ import print_function
import os
import sys
import argparse
import json
from tensorflow.core.framework.graph_pb2 import GraphDef
from tensorflow.python.summary import summary
from tensorflow.python.framework import graph_io
from tensorflow.python.client import session
from tensorflow.python.framework import importer
from tensorflow.python.framework import ops
from tensorflow.python.platform import gfile
from google.protobuf import text_format
from tensorflow_transform.graph_tools import get_dependent_inputs
from tensorflow.python.framework.graph_util import extract_sub_graph
import pickle as pkl
#import ngraph_bridge

class Encapsulates(object):
  helptxt = '''
  usage: import-encapsulates --nodemap <nodemap.pkl> --out <outdir> <file.pbtxt> [<file.pbtxt ...]
  '''
  
  def __init__(self):
    self._args = {}
    self._argv = []
    self._files = {}
    self._graphdef = GraphDef()
    self._modelname = None
    self._modeldirectory = None
    self._nodemap = {}

  # This function controls how errors are handled.
  # For developers/debugging set assert_on_failure to True
  def exit_on_error(self, success, error_message, assert_on_failure=False):
    if not success:
      if assert_on_failure:
        assert success, error_message
      else:
        sys.stderr.write("\n" + error_message + "\n")
        sys.exit(1)
        
  def prepend_to_name(self, graphdef):
    '''
    prepend an extra string to the node name (presumably a scope, to denote encapsulate)
    '''
    self.modify_node_names(
      graphdef, {
        node.name: self._nodemap[node.name] + node.name
        for node in graphdef.node if node.name in self._nodemap
      })

  def sanitize_node_names(self, graphdef):
    '''
    remove '_' from node names. '_' at the beginning of node names indicate internal ops
    which might cause TB to complain
    '''
    self.modify_node_names(
      graphdef, {
        node.name: node.name[1:]
        for node in graphdef.node if node.name[0] == "_"
      })

  def rename_node_name(self, node_name):
    '''
    Given an input to a node in the graph def clean it to find the node name
    '''
    # get rid of caret indicating control edge (^name -> name)
    if node_name.startswith('^'):
      node_name = node_name[1:]
  
    # get rid of output slot (name:0 -> name)
    split_colon = node_name.split(':')
    if len(split_colon) == 1 or len(split_colon) == 2:
      return split_colon[0]
    else:
      self.exit_on_error(False, "Expected node name to have <= 1 colons. " + \
                    "TODO: Handle case with > 1 colons")
      
  def preprocess(self, pbtxt_file):
    if not gfile.Exists(pbtxt_file):
      raise Exception("Input graph file '" + pbtxt_file + "' does not exist!")
    graphdef = GraphDef()

    modifiers = [
      self.prepend_to_name,
      self.sanitize_node_names
    ]
    with open(pbtxt_file, "r") as f:
      protobuf_str = f.read()
      try:
        text_format.Merge(protobuf_str, graphdef)
      except:
        raise Exception("Failed to read pbtxt.")
    for modifier_function in modifiers:
      modifier_function()
    graph_io.write_graph(graphdef,
                        os.path.dirname(pbtxt_file),
                        pbtxt_file,
                        as_text=True)
    

  def process(self, name, pbtxt_file):
    if not gfile.Exists(pbtxt_file):
      raise Exception("Input graph file '" + pbtxt_file + "' does not exist!")
    with open(pbtxt_file, "r") as f:
      protobuf_str = f.read()
      try:
        graphdef = GraphDef()
        text_format.Merge(protobuf_str, graphdef)
        graph_io.write_graph(graphdef,
                             os.path.dirname(pbtxt_file),
                             name + ".pb",
                             as_text=False)
      except:
        raise Exception("Failed to process pbtxt.")


  
  def parse_args(self):
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter,
                                     description=Encapsulates.helptxt)
    parser.add_argument('--outdir',
                        action='store',
                        help="The output directory",
                        type=str,
                        default='tb',
                        required=True)
    parser.add_argument(
      '--nodemap',
      action='store',
      help='Path to nodemap.pkl, which maps attribute paths in declustered files to ngraph clusters',
      type=str,
      default='nodemap.pkl',
      required=True)
    
    self._args, self._argv = parser.parse_known_args()
    with open(self._args.nodemap, 'rb') as nodemap:
      self._nodemap = pkl.load(nodemap)
      parts = os.path.splitext(self._args.nodemap)
      if [[len(parts) == 2]]:
        with open(parts[0]+".json", 'w') as outfile:
          json.dump(self._nodemap, outfile, indent=2)
    self._outdir = os.path.abspath(self._args.outdir)
    if not os.path.exists(self._outdir):
        os.makedirs(self._outdir)
    filename = self._argv[0]
    absfilename = os.path.abspath(filename)
    basename = os.path.basename(absfilename)
    parts = os.path.splitext(basename)
    if [[len(parts) == 2 and parts[1] == ".pbtxt"]]:
      self._modelname = parts[0]
      self._modeldirectory = os.path.dirname(absfilename)
    for filename in self._argv[1:]:
      absfilename = os.path.abspath(filename)
      basename = os.path.basename(absfilename)
      parts = os.path.splitext(basename)
      if [[len(parts) == 2 and parts[1] == ".pbtxt"]]:
        rootname = parts[0]
        dirname = os.path.dirname(absfilename)
        self._files[rootname] = os.path.join(dirname, rootname)
 
  
  def modify_node_names(self, graph_def, node_map):
    '''
    Accepts a graphdef and a map of node name to new node name.
    Replaces the nodes with their new names in the graphdef
    '''
    for node in graph_def.node:
      if node.name in node_map:
        old_name = node.name
        new_name = node_map.get(node.name)
        # print("Replacing: ", node.name, " with ", new_name)
        node.name = new_name
        for _node in graph_def.node:
          for idx, inp_name in enumerate(_node.input):
            # removing the part after ':' in the name
            # removing ^ if present (control dependency)
            colon_split = inp_name.split(':')
            assert len(colon_split) <= 2
            control_dependency_part = '^' if inp_name[0] == '^' else ''
            colon_part = '' if len(
              colon_split) == 1 else ':' + colon_split[1]
            if inp_name.lstrip('^').split(':')[0] == old_name:
              _node.input[idx] = control_dependency_part + \
                                 new_name + colon_part
          # TODO: Do we need to edit this anywhere else other than inputs?

  def get_gdef_from_protobuf(self, pb_filename):
    graph_def = GraphDef()
    if pb_filename.endswith("pbtxt"):
      with open(pb_filename, "r") as f:
        text_format.Merge(f.read(), graph_def)
    else:
      with open(pb_filename, "rb") as f:
        graph_def.ParseFromString(f.read())
    return graph_def
  
  def get_name_type_map(self, graph_def):
    return {n.name: n.op for n in graph_def.node}
  
  def get_possible_output_node_names(self, graph_def):
    '''
    Nodes which do not appear in the inputs of other nodes
    are returned as possible output nodes.
    '''
    nodes_which_appear_at_inputs = set()
    for n in graph_def.node:
      # the list comprehension converts a
      # google.protobuf.pyext._message.RepeatedScalarContainer to a list of strings
      nodes_which_appear_at_inputs.update(
        [self.rename_node_name(i) for i in n.input])
  
    all_node_names = {n.name for n in graph_def.node}
    possible_outputs = all_node_names.difference(nodes_which_appear_at_inputs)
    name_type_map = self.get_name_type_map(graph_def)
    return {k: name_type_map[k] for k in possible_outputs}
  
  def find_encapsulates(self, pbtxt_file):
    if not gfile.Exists(pbtxt_file):
      raise Exception("Input graph file '" + pbtxt_file + "' does not exist!")
    with open(pbtxt_file, "r") as f:
      protobuf_str = f.read()
      try:
        graphdef = GraphDef()
        text_format.Merge(protobuf_str, graphdef)
        #for node in graphdef.node:
        #  if node.op == "NGraphEncapsulate":
          
        #extract_sub_graph(graph)
      except:
        raise Exception("Failed to read pbtxt.")

  
  def pbtxts2pb(self):
    self.preprocess(os.path.join(self._modeldirectory, self._modelname)+".pbtxt")
    for _, path in self._files.items():
      self.preprocess(path+".pbtxt")

    # write out as .pb
    self.process(self._modelname, os.path.join(self._modeldirectory, self._modelname)+".pbtxt")
    for name, path in self._files.items():
      self.process(name, path+".pbtxt")
    return self
  
  #all_tensors = [tensor for op in sess.graph.get_operations() for tensor in op.values()]
  def pb2tb(self, pbfile, outdir):
    """
    pb2tb
      parameters: pbfile, outdir
    """
    with session.Session(graph=ops.Graph()) as sess:
      with sess.graph.as_default():
        graphdef = GraphDef()
        with gfile.GFile(pbfile, 'rb') as f:
          graphdef.ParseFromString(f.read())
        importer.import_graph_def(graphdef)
        absoutdir = os.path.abspath(outdir)
        if not os.path.exists(absoutdir):
          os.makedirs(absoutdir)
        pb_visual_writer = summary.FileWriter(absoutdir)
        pb_visual_writer.add_graph(sess.graph)

  def pb2pbtxt(self, pbfile):
    """
    pb2pbtxt
      parameters: pbfile
    """
    try:
      graphdef = GraphDef()
      absfilename = os.path.abspath(pbfile)
      basename = os.path.basename(absfilename)
      dirname = os.path.dirname(absfilename)
      parts = os.path.splitext(basename)
      if not gfile.Exists(absfilename):
        raise Exception("Input graph file '" + absfilename + "' does not exist")
      with gfile.FastGFile(absfilename, 'rb') as f:
        graphdef.ParseFromString(f.read())
      graph_io.write_graph(graphdef, dirname, parts[0]+".pbtxt", as_text=True)
    except:
      raise Exception("Failed to process pb.")


  def pbtxt2pb(self, pbtxtfile):
    """
    pbtxt2pb 
      parameters: pbtxtfile
    """
    if not gfile.Exists(pbtxtfile):
      raise Exception("Input graph file '" + pbtxtfile + "' does not exist")
    with open(pbtxtfile, "r") as f:
      protobuf_str = f.read()
      try:
        absfilename = os.path.abspath(pbtxtfile)
        basename = os.path.basename(absfilename)
        dirname = os.path.dirname(absfilename)
        parts = os.path.splitext(basename)
        if [[len(parts) == 2]]:
          graphdef = GraphDef()
          text_format.Merge(protobuf_str, graphdef)
          graph_io.write_graph(graphdef, dirname, parts[0]+".pb", as_text=False)
      except:
        raise Exception("Failed to process pbtxt.")

def main():
  encapsulates = Encapsulates()
  encapsulates.parse_args()
  encapsulates.pbtxts2pb()
  
if __name__ == '__main__':
  main()
