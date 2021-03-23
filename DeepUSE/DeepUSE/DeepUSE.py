import os
import unittest
import logging
import vtk, qt, ctk, slicer
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin

import numpy as np
import time
import torch
import argparse
import time
import math

from models.network import USENet, ReUSENet
from utils import parse_configuration, to_bmode, get_strain, warp_image

class DeepUSE(ScriptedLoadableModule):
  """Uses ScriptedLoadableModule base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent):
    ScriptedLoadableModule.__init__(self, parent)
    self.parent.title = "DeepUSE" 
    self.parent.categories = ["DeepUSE"]
    self.parent.dependencies = []
    self.parent.contributors = ["Remi Delaunay (University College London, King's College London)"]
    # TODO: update with short description of the module and a link to online module documentation
    self.parent.helpText = """
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#DeepUSE">module documentation</a>.
"""
    self.parent.acknowledgementText = """
This file was originally developed by Remi Delaunay from University College London & King's College London.
"""
    # Additional initialization step after application startup is complete
    slicer.app.connect("startupCompleted()", registerSampleData)

def registerSampleData():
  """
  Add data sets to Sample Data module.
  """
  # It is always recommended to provide sample data for users to make it easy to try the module,
  # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

  import SampleData
  iconsPath = os.path.join(os.path.dirname(__file__), 'Resources/Icons')

  # To ensure that the source code repository remains small (can be downloaded and installed quickly)
  # it is recommended to store data sets that are larger than a few MB in a Github release.
  SampleData.SampleDataLogic.registerCustomSampleDataSource(
      category='data',
      sampleName='rf_data1',
      uris='https://github.com/RemiDelaunay/DeepUSE_TestData/raw/0e5b4f8aa5d5ef4c2815f7f3106e0f24a7c6434e/TestData/rf0299_07.nrrd',
      fileNames='rf0299_07.nrrd',
      nodeNames='rf0299_07',
      loadFileType='VolumeFile',
      thumbnailFileName=os.path.join(iconsPath, 'rf0299_07.png')
      )

class DeepUSEWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
  """Uses ScriptedLoadableModuleWidget base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self, parent=None):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.__init__(self, parent)
    VTKObservationMixin.__init__(self)  # needed for parameter node observation
    self.logic = None
    self._parameterNode = None
    self._updatingGUIFromParameterNode = False

  def setup(self):
    """
    Called when the user opens the module the first time and the widget is initialized.
    """
    ScriptedLoadableModuleWidget.setup(self)

    # Load widget from .ui file (created by Qt Designer).
    # Additional widgets can be instantiated manually and added to self.layout.
    uiWidget = slicer.util.loadUI(self.resourcePath('UI/DeepUSE.ui'))
    self.layout.addWidget(uiWidget)
    self.ui = slicer.util.childWidgetVariables(uiWidget)

    # Set scene in MRML widgets. Make sure that in Qt designer the top-level qMRMLWidget's
    # "mrmlSceneChanged(vtkMRMLScene*)" signal in is connected to each MRML widget's.
    # "setMRMLScene(vtkMRMLScene*)" slot.
    uiWidget.setMRMLScene(slicer.mrmlScene)

    # Create logic class. Logic implements all computations that should be possible to run
    # in batch mode, without a graphical user interface.
    self.logic = DeepUSELogic()

    # Connections

    # These connections ensure that we update parameter node when scene is closed
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
    self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

    # These connections ensure that whenever user changes some settings on the GUI, that is saved in the MRML scene
    # (in the selected parameter node).
    self.ui.PathLineConfig.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.lineEditHost.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.spinBoxPort.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.spinBoxInterframe.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.connectButton.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.inputSelector.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.inferenceButton.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)
    self.ui.checkBox.connect("currentNodeChanged(vtkMRMLNode*)", self.updateParameterNodeFromGUI)

    # Buttons
    self.ui.connectButton.connect('clicked(bool)', self.onConnectButton)
    self.ui.loadButton.connect('clicked(bool)', self.onLoadButton)
    self.ui.inferenceButton.connect('clicked(bool)', self.onInferenceButton)
    # Make sure parameter node is initialized (needed for module reload)
    self.initializeParameterNode()

  def cleanup(self):
    """
    Called when the application closes and the module widget is destroyed.
    """
    self.removeObservers()

  def enter(self):
    """
    Called each time the user opens this module.
    """
    # Make sure parameter node exists and observed
    self.initializeParameterNode()

  def exit(self):
    """
    Called each time the user opens a different module.
    """
    # Do not react to parameter node changes (GUI wlil be updated when the user enters into the module)
    self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

  def onSceneStartClose(self, caller, event):
    """
    Called just before the scene is closed.
    """
    # Parameter node will be reset, do not use it anymore
    self.setParameterNode(None)

  def onSceneEndClose(self, caller, event):
    """
    Called just after the scene is closed.
    """
    # If this module is shown while the scene is closed then recreate a new parameter node immediately
    if self.parent.isEntered:
      self.initializeParameterNode()

  def initializeParameterNode(self):
    """
    Ensure parameter node exists and observed.
    """
    # Parameter node stores all user choices in parameter values, node selections, etc.
    # so that when the scene is saved and reloaded, these settings are restored.

    self.setParameterNode(self.logic.getParameterNode())

    #Select default input nodes if nothing is selected yet to save a few clicks for the user
    if not self._parameterNode.GetNodeReference("InputVolume"):
      firstVolumeNode = slicer.mrmlScene.GetFirstNodeByClass("vtkMRMLScalarVolumeNode")
      if firstVolumeNode:
        self._parameterNode.SetNodeReferenceID("InputVolume", firstVolumeNode.GetID())

  def setParameterNode(self, inputParameterNode):
    """
    Set and observe parameter node.
    Observation is needed because when the parameter node is changed then the GUI must be updated immediately.
    """

    if inputParameterNode:
      self.logic.setDefaultParameters(inputParameterNode)

    # Unobserve previously selected parameter node and add an observer to the newly selected.
    # Changes of parameter node are observed so that whenever parameters are changed by a script or any other module
    # those are reflected immediately in the GUI.
    if self._parameterNode is not None:
      self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)
    self._parameterNode = inputParameterNode
    if self._parameterNode is not None:
      self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self.updateGUIFromParameterNode)

    # Initial GUI update
    self.updateGUIFromParameterNode()

  def updateGUIFromParameterNode(self, caller=None, event=None):
    """
    This method is called whenever parameter node is changed.
    The module GUI is updated to show the current state of the parameter node.
    """
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    # Make sure GUI changes do not call updateParameterNodeFromGUI (it could cause infinite loop)
    self._updatingGUIFromParameterNode = True

    # Update node selectors and sliders
    self.ui.PathLineConfig.setCurrentPath(self._parameterNode.GetParameter("PathConfig"))
    self.ui.lineEditHost.text = self._parameterNode.GetParameter("Host")
    self.ui.spinBoxPort.value = int(self._parameterNode.GetParameter("Port"))
    self.ui.spinBoxInterframe.value = int(self._parameterNode.GetParameter("Interframe"))
    self.ui.inputSelector.setCurrentNode(self._parameterNode.GetNodeReference("InputVolume"))

    # All the GUI updates are done
    self._updatingGUIFromParameterNode = False

  def updateParameterNodeFromGUI(self, caller=None, event=None):
    """
    This method is called when the user makes any change in the GUI.
    The changes are saved into the parameter node (so that they are restored when the scene is saved and loaded).
    """
    if self._parameterNode is None or self._updatingGUIFromParameterNode:
      return

    wasModified = self._parameterNode.StartModify()  # Modify all properties in a single batch

    self._parameterNode.SetNodeReferenceID("InputVolume", self.ui.inputSelector.currentNodeID)

    self._parameterNode.EndModify(wasModified)

  def onConnectButton(self):
    """
    """
    if self.logic.cnode:
      slicer.mrmlScene.RemoveNode(self.logic.cnode)
    self.logic.connect(self.ui.lineEditHost.text, self.ui.spinBoxPort.value)
    time.sleep(1)
    if self.logic.is_connected():
      logging.info('OpenIGTLink connection successful.')

  def onLoadButton(self):
    """
    """
    configuration = parse_configuration(self.ui.PathLineConfig.currentPath)["model_params"]
    self.logic.load_network(configuration)

  def onInferenceButton(self):
    """
    """
    import time
    if self.logic.cnode and self.logic.is_connected() and self.logic.is_incomingMRMLNode(self.ui.inputSelector.currentNode()):
      logging.info('Online inference on input node {} have started'.format(self.ui.inputSelector.currentNode().GetName()))
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.spinBoxInterframe.value, store_results=self.ui.checkBox.checkState(), online=True)
    else:
      logging.info('Offline inference on input node {} have started'.format(self.ui.inputSelector.currentNode().GetName()))
      start_time = time.time()
      self.logic.process(self.ui.inputSelector.currentNode(), self.ui.spinBoxInterframe.value, store_results=self.ui.checkBox.checkState(), online=False)
      logging.info("Processing time was performed in {0:04} seconds".format(round(time.time() - start_time,4)))

# DeepUSELogic---------------------------

class DeepUSELogic(ScriptedLoadableModuleLogic):
  """This class should implement all the actual
  computation done by your module.  The interface
  should be such that other python code can import
  this class and make use of the functionality without
  requiring an instance of the Widget.
  Uses ScriptedLoadableModuleLogic base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def __init__(self):
    """
    Called when the logic class is instantiated. Can be used for initializing member variables.
    """
    ScriptedLoadableModuleLogic.__init__(self)
    self.InputVolume = None
    self.cnode = None

  def load_network(self, configuration):
    self.network_name = configuration["model_name"]
    self.use_cuda = torch.cuda.is_available()
    self.device = torch.device('cuda:0') if self.use_cuda else torch.device('cpu')
    if self.network_name == "ReUSENet":
      self.net = ReUSENet(in_channel=2, num_channel_initial=configuration['num_channel_initial'])
      logging.info('Loading ReUSENet')
    elif self.network_name == "USENet":
      self.net = USENet(in_channel=2, num_channel_initial=configuration['num_channel_initial'])
      logging.info('Loading USENet')
    else:
      logging.info('Please select an existing network (reusenet, usenet)')

    load_filename = '{0}_net_{1}.pth'.format(configuration["load_checkpoint"], configuration["model_name"])
    load_path = os.path.join(configuration["checkpoint_path"], load_filename)
    state_dict = torch.load(load_path, map_location=self.device)
    self.net.load_state_dict(state_dict)
    self.net = self.net.to(self.device)
    #self.model.eval()
  
  def connect(self, hostname, port):
    self.cnode = slicer.vtkMRMLIGTLConnectorNode()
    slicer.mrmlScene.AddNode(self.cnode)
    self.cnode.SetName('IGTLConnector')
    self.cnode.SetTypeClient(hostname, port)
    self.cnode.Start()
  
  def is_connected(self):
    """Check whether an IGT connection has been successful.
    :param connector: A vtkMRMLIGTConnectorNode instance.
    :return: True if the node's status indicates it is connected, else False.
    """
    try:
        connected_state = slicer.vtkMRMLIGTLConnectorNode.StateConnected
    except AttributeError:
        connected_state = slicer.vtkMRMLIGTLConnectorNode.STATE_CONNECTED
    return self.cnode.GetState() == connected_state

  def is_incomingMRMLNode(self, volumeNode):
    node_list = []
    for i in range(self.cnode.GetNumberOfIncomingMRMLNodes()):
      node_list += [self.cnode.GetIncomingMRMLNode(i).GetID()]
    if any(volumeNode.GetID() in s for s in node_list):
      return True
    else:
      return False

  def createBmodeNode(self, volume):
      self.BmodeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", '{}_Bmode'.format(volume.GetName()))
      self.BmodeNode.SetSpacing(0.2,0.02,1)
      self.BmodeNode.SetIJKToRASDirections(-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0)

  def createStrainNode(self, volume):
      self.StrainNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", '{}_Strain'.format(volume.GetName()))
      self.StrainNode.SetSpacing(0.2,0.02,1)
      self.StrainNode.SetIJKToRASDirections(-1.0,0.0,0.0,0.0,-1.0,0.0,0.0,0.0,1.0)

  # Callback function that will be called each time the transform is modified
  def onMyImageModified(self, caller, event):
      if isinstance(self.fixed,np.ndarray):
          self.moving = slicer.util.arrayFromVolume(self.InputVolume).copy()
          self.moving = self.moving[:,100:-300,:128] # to fit network's input size
          self.input = torch.from_numpy(np.concatenate([self.fixed,self.moving],axis=0)).unsqueeze(0)
          self.Strain = self.inference(self.input)
          self.Bmode = np.expand_dims(to_bmode(self.fixed[:,143:-143,10:-10]),0) # image cropped to fit strain ROI
          if self.store_results:
            try:
              slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.StrainNode.GetID()), np.concatenate([slicer.util.arrayFromVolume(self.StrainNode).copy(),self.Strain],axis=0))
              slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.BmodeNode.GetID()), np.concatenate([slicer.util.arrayFromVolume(self.BmodeNode).copy(),self.Bmode],axis=0))
            except AttributeError:
              slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.StrainNode.GetID()), self.Strain)
              slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.BmodeNode.GetID()), self.Bmode)
          else:
            slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.StrainNode.GetID()), self.Strain)
            slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.BmodeNode.GetID()), self.Bmode)
          # Assign bmode to red slice
          red_logic = slicer.app.layoutManager().sliceWidget("Red").sliceLogic()
          red_logic.GetSliceCompositeNode().SetBackgroundVolumeID(self.BmodeNode.GetID())
          red_logic.GetSliceNode().SetOrientationToAxial()
          red_logic.FitSliceToAll()
          # Assign strain to yellow slice
          yellow_logic = slicer.app.layoutManager().sliceWidget("Yellow").sliceLogic()
          yellow_logic.GetSliceCompositeNode().SetBackgroundVolumeID(self.StrainNode.GetID())
          yellow_logic.GetSliceNode().SetOrientationToAxial()
          yellow_logic.FitSliceToAll()
          # slicer.util.setSliceViewerLayers(background=self.StrainNode, fit=True)
          if self.StrainNode.GetDisplayNode():
            self.StrainNode.GetDisplayNode().SetAndObserveColorNodeID('vtkMRMLColorTableNodeFileViridis.txt')

          self.processed += 1
          if self.processed == self.interframe:
            self.processed = 0
            self.prev_state = [None,None,None,None,None]
            self.fixed = self.moving.copy()
      else:
          self.processed = 0
          self.prev_state = [None,None,None,None,None]
          self.fixed = slicer.util.arrayFromVolume(self.InputVolume).copy()
          self.fixed = self.fixed[:,100:-300,:128]

  def add_observer(self, node, func):
      return node.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, func)
    
  def inference(self, image):
      image = image.to(self.device)
      if self.network_name == 'ReUSENet':
        with torch.no_grad():
          self.disp_map, self.prev_state = self.net(image, self.prev_state)
      else:
        with torch.no_grad():
          self.disp_map = self.net(image)
      strain = get_strain(self.disp_map[:, 1:2, :, :])
      strain_compensated = warp_image(strain,self.disp_map)
      return strain_compensated[0,:,143:-143,10:-10].cpu().numpy()

  def process(self, image_node, interframe, store_results=True, online=False):
    self.interframe = interframe
    self.createBmodeNode(image_node)
    self.createStrainNode(image_node)
    self.fixed = None
    self.processed = 0
    self.store_results = store_results
    self.online = online
    if self.online == False:
      sequence = slicer.util.arrayFromVolume(image_node).copy()
      self.InputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", 'tmpVolume')
      slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.InputVolume.GetID()), sequence[0:1,...])
      self.InputVolume.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onMyImageModified)
      for i in range(0,len(sequence)):
        slicer.util.updateVolumeFromArray(slicer.mrmlScene.GetNodeByID(self.InputVolume.GetID()), sequence[i:i+1,...])
      slicer.mrmlScene.RemoveNode(slicer.mrmlScene.GetNodeByID(self.InputVolume.GetID()))      
    else:
      self.InputVolume = image_node
      self.InputVolume.AddObserver(slicer.vtkMRMLVolumeNode.ImageDataModifiedEvent, self.onMyImageModified)

  def setDefaultParameters(self, parameterNode):
    """
    Initialize parameter node with default settings.
    """
    if not parameterNode.GetParameter("Port"):
      parameterNode.SetParameter("Port", "18944")
    if not parameterNode.GetParameter("Host"):
      parameterNode.SetParameter("Host", "localhost")
    if not parameterNode.GetParameter("Interframe"):
      parameterNode.SetParameter("Interframe", "1")

#
# DeepUSETest
#

class DeepUSETest(ScriptedLoadableModuleTest):
  """
  This is the test case for your scripted module.
  Uses ScriptedLoadableModuleTest base class, available at:
  https://github.com/Slicer/Slicer/blob/master/Base/Python/slicer/ScriptedLoadableModule.py
  """

  def setUp(self):
    """ Do whatever is needed to reset the state - typically a scene clear will be enough.
    """
    slicer.mrmlScene.Clear()
  
  def download(self, url: str, dest_folder: str):
    import requests
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)  # create folder if it does not exist

    filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
    file_path = os.path.join(dest_folder, filename)

    r = requests.get(url, stream=True)
    if r.ok:
        print("saving to", os.path.abspath(file_path))
        with open(file_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024 * 8):
                if chunk:
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())
    else: 
        print("Download failed: status code {}\n{}".format(r.status_code, r.text))

  def runTest(self):
    """Run as few or as many tests as needed here.
    """
    self.setUp()
    self.test_DeepUSE1()

  def test_DeepUSE1(self):
    """ Ideally you should have several levels of tests.  At the lowest level
    tests should exercise the functionality of the logic with different inputs
    (both valid and invalid).  At higher levels your tests should emulate the
    way the user would interact with your code and confirm that it still works
    the way you intended.
    One of the most important features of the tests is that it should alert other
    developers when their changes will have an impact on the behavior of your
    module.  For example, if a developer removes a feature that you depend on,
    your test should break so they know that the feature is needed.
    """

    self.delayDisplay("Starting the test")

    # Get/create input data

    import SampleData
    import time
    registerSampleData()
    inputVolume = SampleData.downloadSample('rf_data1')
    dest_folder = 'test_network_weights'
    self.download('https://github.com/RemiDelaunay/DeepUSE_TestData/raw/main/TestData/300_net_ReUSENet.pth',dest_folder)
    self.delayDisplay('Loaded test data set')

    logic = DeepUSELogic()

    network_configuration = {
        "model_name": "ReUSENet",
        "num_channel_initial":8,
        "checkpoint_path": os.path.abspath(dest_folder),
        "load_checkpoint": 300
    }

    logic.load_network(network_configuration)
    
    start_time = time.time()
    logic.process(inputVolume, 1, store_results=True, online=False)
    self.delayDisplay("Processing time was {0:05} seconds".format(time.time() - start_time))


