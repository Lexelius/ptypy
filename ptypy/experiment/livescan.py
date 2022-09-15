"""
Copied LiveScan(PtyScan) from local file 'v4_livescan_inprogress.py'

Current state: In progress, not checked to work yet.
                Connects to directly to streamer



Old notes:
----------------------------------------------------------------------------
Things to fix / troubleshooting notes:
----------------------------------------------------------------------------
* Could you maybe add a weight to the iterations somehow so that the reconstructions from
    the earlier iterations (which are calculated with less frames) contributes with a
    lesser impact, or would this be impossible due to the iterative nature of the reconstruction?
* If num_iter have been reached before all patterns have been collected then
    Ptycho stops because it thinks it's finished!
    -> Define a number of iterations for recon that starts counting only after scan is over: yes!
* If 'p.scans.contrast.data.shape = 128' is included in the 'livescan.py' script then
    ptycho-reconstruction becomes super fast and stops before scan is over as above.
* Check: how to specify " kind = 'full_flat' " as input for save_run(..) that is called by Ptycho.
    Or rather, is there a way to save the pods to the .ptyr files as well?
* Iterations will not be performed while ptycho.model.new_data() has new data, meaning that if data is
    streamed seemingly continuous, Ptycho won't start with the iterations until all data has been acquired..
*Automatically check which iteration has the lowes error and chose that reconstruction as the final reconstruction instead of just the last one.

* Number of frames included in iteration  0    10    20    30  40  50  60  70  80  90  100
    min_frames = 10, DM.numiter = 10:
        check return:                     6*   8*9    0*
        latest_pos_index_received         27* 69*105 120*
        Repackaged data from frame        21* 61*96  120*
        .ptyr / error_local                -   21    96   120

    min_frames = 1, DM.numiter = 1:
        check return:                     0     0
        latest_pos_index_received         28    120
        Repackaged data from frame        28    120
        .ptyr / error_local               -     120
    min_frames = 1, DM.numiter = 10:
        check return:                     0     0
        latest_pos_index_received         29    120
        Repackaged data from frame        29    120
        .ptyr / error_local
 
SOLVED PROBLEMS:
✖✖✖✖✖✖✖✖✖ Ptycho keeps going in to LiveScan.check() after all frames have been acquired, which overwrites
            then self.end_of_scan..
            ✖ SOLUTION: Move 'self.end_of_scan = False' under LiveScan.init() instead of under LiveScan.check().
✖✖✖✖✖✖✖✖✖ There is still a small difference in the resulting object and exit wave compared to original script.
            Exit waves, masks, object etc get their values already during ptycho level 2 ( P.init_data() )!
            While the object is still just a uniform matrix at this point, the value it
            is filled with differs between livescan and 1222!
            - P.probe, Pobj, Pexit gets their first view and storage in line 981 of 'manager.py' (during P2) but are
                still just a zero matrix at this point.
            - P.probe is filled with values at line 1122 of 'manager.py'
            ✖ SOLUTION: Difference occurs in line359 of sample.py and comes from different precisions in variable k,
                which inherits the type from the energy! In 1222_...py energy gets read into a ndarray of
                size (1,) with type float64, whereas my energy is loaded directly as a float, i.e. float32!!
✖✖✖✖✖✖✖✖✖ Check if there is a way to see how many pods/frames that are included in each .ptyr file.
            ✖ SOLUTION: Using inspect.getouterframes(..) to retrieve variables from classes/methods/functions that
                calls on LiveScan. This is implemented in my fynction "backtrace()"
"""

"""
To do:
* Check how the time for loading/Repacking data changes with the number of frames
-----------------------------------------------------------
Notes:
-----------------------------------------------------------
Subclasses of PtyScan can be made to override to tweak the methods of base class PtyScan.
Methods defined in PtyScan(object) are:
    ¨def __init__(self, pars=None, **kwargs):
    def initialize(self):
    ¨def _finalize(self):
    ^def load_weight(self):
    ^def load_positions(self):
    ^def load_common(self):
    def post_initialize(self):
    ¨def _mpi_check(self, chunksize, start=None):
    ¨def _mpi_indices(self, start, step):
    def get_data_chunk(self, chunksize, start=None):
    def auto(self, frames):
    ¨def _make_data_package(self, chunk):
    ¨def _mpi_pipeline_with_dictionaries(self, indices):
    ^def check(self, frames=None, start=None):
    ^def load(self, indices):
    ^def correct(self, raw, weights, common):
    ¨def _mpi_autocenter(self, data, weights):
    def report(self, what=None, shout=True):
    ¨def _mpi_save_chunk(self, kind='link', chunk=None):

¨: Method is protected (or private if prefix is __).
^: Description explicitly says **Override in subclass for custom implementation**.
"""

import numpy as np
import zmq
from zmq.utils import jsonapi as json
import time
import bitshuffle
import struct
import ptypy
from ptypy.core import Ptycho
from ptypy.core.data import PtyScan
from ptypy import utils as u
from ptypy.utils import parallel
from ptypy import defaults_tree
from ptypy.experiment import register
from ptypy.utils.verbose import headerline
import inspect

logger = u.verbose.logger
def logger_info(*arg):
    """
    Just an alternative to commenting away logger messages.
    """
    return


def decompress(data, shape, dtype):
    """
    Copied from nanomax_streaming.py, decompresses streamed images when
    recv_multipart(flags=zmq.NOBLOCK) is used
    """
    block_size = struct.unpack('>I', data[8:12])[0] // dtype.itemsize
    data = np.frombuffer(data[12:], np.int8)
    output = bitshuffle.decompress_lz4(arr=data,
                                       shape=shape,
                                       dtype=dtype,
                                       block_size=block_size)
    return output


def backtrace(fname=None):
    """
    Retrieves variables from functions/classes which are calling LiveScan,
    such that current nr of pods and the nr of currently performed iterations
    can be printed.
    """
    logger.info(headerline('', 'c', '°'))
    curframe = inspect.currentframe()
    calframe = inspect.getouterframes(curframe, 2)
    #### print('calframe.__len__() = %d' % calframe.__len__())
    #### try:
    ####     for fr in range(0, calframe.__len__()):
    ####         print(f'calframe[{fr}] = ', calframe[fr])
    ####         print(f'calframe[{fr}][0].f_locals.keys() = \n', calframe[fr][0].f_locals.keys(), '\n')
    #### except:
    ####     pass
    #print(*[(frame.lineno, frame.filename) for frame in calframe], sep="\n")
    try:
        ##OnMyMac## ptychoframe = [frame.filename for frame in calframe].index('/opt/anaconda3/envs/Contrast_PtyPyLive_v1/lib/python3.7/site-packages/ptypy/core/ptycho.py') ## or '/opt/anaconda3/envs/ptypy_contrast_NM-utils/lib/python3.7/site-packages/ptypy/core/ptycho.py'   or  '/opt/anaconda3/envs/Contrast_PtyPyLive_v1/lib/python3.7/site-packages/ptypy/core/ptycho.py'   or  '/Users/lexelius/Documents/PtyPy/ptypy-master/ptypy/core/ptycho.py'
        ptychoframe = [frame.filename for frame in calframe].index(ptypy.core.ptycho.__file__)
        ptycho_self = calframe[ptychoframe][0].f_locals['self']
        #### ptycho_self.print_stats()
        active_pods = sum(1 for pod in ptycho_self.pods.values() if pod.active)
        all_pods = len(ptycho_self.pods.values())
        print(f'---- Total Pods {all_pods} ({active_pods} active) ----')
    except:
        pass
    if calframe[7].function == 'run':# and calframe[7].lineno == 632:
        ptycho_engine = calframe[7][0].f_locals['engine']
        print('ptycho_engine.curiter = ', ptycho_engine.curiter)
    if fname != None:
        with open(fname, 'a') as f:
            f.write(f'Total Pods: {all_pods} ({active_pods} active), \n')
            try:
                f.write(f'Iteration:  {ptycho_engine.curiter}, \n\n')
            except:
                pass
    logger.info(headerline('', 'c', '°') + '\n')


# ordinal = lambda n: "%d%s" % (n,"tsnrhtdd"[(n//10%10!=1)*(n%10<4)*n%10::4])
# print([ordinal(n) for n in range(1,32)])

##@defaults_tree.parse_doc('scandata.LiveScan')
@register()
class LiveScan(PtyScan):
    """
    A PtyScan subclass to extract data from a numpy array.

    Defaults:

    [name]
    type = str
    default = LiveScan
    help =

    [host]
    default = 'tcp://b-nanomax-controlroom-cc-3'
    type = str
    help = Name of the publishing host
    doc =  contrast

    [port]
    default = 5556
    type = int
    help = Port number on the publishing host
    doc = # tcp://b-nanomax-controlroom-cc-3:5556 eller om det inte funkar e.g på clustret: 172.16.125.30 (for motor daata stuff)

    [xMotor]
    default = sx
    type = str
    help = Which x motor to use
    doc =

    [yMotor]
    default = sy
    type = str
    help = Which y motor to use
    doc =

    [detector]
    default = 'diff'
    type = str
    help = Which detector from the contrast stream to use

    [detector_host]
    default = "tcp://b-daq-node-2"
    type = str
    help = Take images from a separate stream - hostname
    doc = streaming-reciever at gitlab, info available at https://wiki.maxiv.lu.se/index.php/NanoMAX:Dectris_-_Eiger2_X4M ...host and port for detector is the "secondary zmq port for arbitrary processing"

    [detector_port]
    default = 20001
    type = int
    help = Take images from a separate stream - port
    doc = Default is the port used by NanoMax Eiger4M

    [block_wait_count]
    default = 0
    type = int
    help = Signals a WAIT to the model after this many blocks.
    """

    def __init__(self, pars=None, **kwargs):

        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        p = self.DEFAULT.copy(depth=99)
        p.update(pars)
        p.update(kwargs)

        super(LiveScan, self).__init__(p, **kwargs)
        self.context = zmq.Context()

        # main socket, reporting positions and motor stuff from contrast
        socket = self.context.socket(zmq.SUB)
        socket.connect("%s:%u" % (self.info.host, self.info.port))
        socket.setsockopt(zmq.SUBSCRIBE, b"")  # subscribe to all topics
        self.socket = socket

        # separate detector socket
        self.stream_images = None not in (self.info.detector_host,
                                          self.info.detector_port)
        if self.stream_images and parallel.master:
            det_socket = self.context.socket(zmq.PULL)
            det_socket.connect("%s:%u" % (self.info.detector_host, self.info.detector_port))
            self.det_socket = det_socket

        self.latest_pos_index_received = -1 ## ToDo: check if these 2 should be in the beginning of check() instead
        self.latest_det_index_received = -1
        self.incoming = {}
        self.incoming_det = {}
        self.end_of_scan = False
        self.end_of_det_stream = False ## ToDo: check if this one is neccesary to implement
        self.checknr = 0
        self.loadnr = 0
        self.checktottime = 0
        self.loadtottime = 0
        self.t = time.gmtime()
        self.t = f'{self.t[0]}-{self.t[1]:02d}-{self.t[2]:02d}__{self.t[3]:02d}-{self.t[4]:02d}'
        #backtrace()

        self.p = p
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().init()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')


    def check(self, frames=None, start=None):
        """
        Only called on the master node.
        """
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        t0 = time.perf_counter()
        self.checknr += 1
        logger.info('check() has now been called %d times.' % self.checknr)  ##
        ##OnMyMac## backtrace(f'/Users/lexelius/Documents/Contrast/contrast-master/experimental-data/livescan_output/poditstats_{self.t}.txt')
        backtrace(f'/home/reblex/Documents/Reconstructions/livescan_output/poditstats_{self.t}.txt')

        # To pause the data acquiring in hope of starting iterations earlier
        bwc = self.p.block_wait_count
        if bwc >= 1 and self.checknr % (bwc + 1) == 0:  ## ToDo: Make a better solution than using self.checknr
            frames_accessible = 0
            return frames_accessible, self.end_of_scan


        # get all frames from the main socket
        while True:
            try:
                msg = self.socket.recv_pyobj(flags=zmq.NOBLOCK)  ## NOBLOCK returns None if a message is not ready
                logger.info('######## Received a message')  ##
                ##headers = ('path' in msg.keys())
                ##emptymsg = ('heartbeat' in msg.values()) # just a message from contrast to keep connection alive
                ######if 'running' in msg.values():  # if zmq did not send a path: then save this message
                if msg['status'] == 'started':
                    self.meta.energy = np.float64([msg['snapshot']['energy']]) * 1e-3 ## Read energy from beamline snapshot
                    logger.info('############ RecorderHeader received; SCAN STARTING!')  ##
                elif msg['status'] == 'running':
                    self.latest_pos_index_received += 1
                    self.incoming[self.latest_pos_index_received] = msg
                    logger.info('############ Frame nr. %d received' % self.latest_pos_index_received)  ##
                    break ## include this break if you want to start iterations befora all frames have been acquired
                elif msg['status'] == 'finished': ## 'msgEOS':  # self.EOS:
                    self.end_of_scan = True
                    logger.info('############ RecorderFooter received; END OF SCAN!')  ##
                    break
                else:
                    logger.info('############ Message was not important')  ##
            except zmq.ZMQError:
                logger.info('######## Waiting for messages')  #w2#
                # no more data available - working around bug in ptypy here
                if self.latest_pos_index_received < self.info.min_frames * parallel.size:
                    logger.info('############ self.latest_pos_index_received = %u , self.info.min_frames = %d , parallel.size = %d' % (self.latest_pos_index_received, self.info.min_frames, parallel.size))  ##
                    logger.info('############ Not enough frames received, have %u frames, waiting...' % (self.latest_pos_index_received + 1))
                    time.sleep(1)
                else:
                    logger.info('############ Will process gathered data')  ##
                    break

        # get all frames from the detector socket
        while self.stream_images:
            try:
                parts = self.det_socket.recv_multipart(flags=zmq.NOBLOCK)
                info = json.loads(parts[0])
                shape = info['shape']
                dtype = np.dtype(info['type'])
                img = decompress(parts[1], shape, dtype)
                self.latest_det_index_received += 1
                self.incoming_det[self.latest_det_index_received] = img

            except zmq.ZMQError:
                # no more data available - working around bug in ptypy here
                if self.latest_det_index_received < self.info.min_frames * parallel.size:
                    logger.info('have %u detector frames, waiting...' % (self.latest_det_index_received + 1))
                    time.sleep(.5)
                else:
                    break

        ind = self.latest_pos_index_received
        logger.info('#### latest_pos_index_received = %d' % ind)
        if self.stream_images:
            ind = min(ind, self.latest_det_index_received)
            logger.info('#### latest_det_index_received = %d' % self.latest_det_index_received)
        logger.info('#### check return (ind - start + 1) = %d, self.end_of_scan = %d' % ((ind - start + 1), self.end_of_scan))
        t1 = time.perf_counter()
        self.checktottime += t1-t0
        logger.info('#### Time spent in check = %f, accumulated time = %f' % ((t1-t0), self.checktottime))
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().check()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')
        return (ind - start + 1), (self.end_of_scan and self.end_of_det_stream)

    def load(self, indices):
        # indices are generated by PtyScan's _mpi_indices method.
        # It is a diffraction data index lists that determine
        # which node contains which data.
        raw, weight, pos = {}, {}, {}
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Entering LiveScan().load()', 'c', '#'))
        logger.info(headerline('', 'c', '#'))
        t0 = time.perf_counter()
        self.loadnr += 1
        logger.info('load() has now been called %d times.' % self.loadnr)  ##
        ##OnMyMac## backtrace(f'/Users/lexelius/Documents/Contrast/contrast-master/experimental-data/livescan_output/poditstats_{self.t}.txt')
        backtrace(f'/home/reblex/Documents/Reconstructions/livescan_output/poditstats_{self.t}.txt')

        # communication
        if parallel.master:
            # send data to each node
            for node in range(1, parallel.size):
                node_inds = parallel.receive(source=node)
                dct = {i:self.incoming[i] for i in node_inds}
                if self.stream_images: ## added fo separate streams
                    # merge streamed images
                    for i in node_inds: ## added fo separate streams
                        dct[i][self.info.detector] = self.incoming_det[i] ## added fo separate streams
                parallel.send(dct, dest=node)
                for i in node_inds:
                    del self.incoming[i]
                    if self.stream_images: ## added fo separate streams
                        del self.incoming_det[i] ## added fo separate streams

            # take data for this node
            dct = {i: self.incoming[i] for i in indices}
            if self.stream_images:
                # merge streamed images
                for i in indices:
                    dct[i][self.info.detector] = self.incoming_det[i]
            for i in indices:
                del self.incoming[i]
                if self.stream_images:
                    del self.incoming_det[i]
        else:
            # receive data from the master node
            parallel.send(indices, dest=0)
            dct = parallel.receive(source=0)
        #logger_info(dct)

        # repackage data and return
        for i in indices:
            try:
                ## raw[i] = dct[i]['diff'] ## used for single stream
#                ToDo
#                      - implement this
                raw[i] = dct[i][self.info.detector]
                #            pos[i] = np.array([
                #                        dct[i][self.info.xMotor],
                #                        dct[i][self.info.yMotor],
                #                        ]) * 1e-6
                x = dct[i][self.info.xMotor]
                y = dct[i][self.info.yMotor]
                ## x = dct[i]['x'] ## used for single stream
                ## y = dct[i]['y'] ## used for single stream
                pos[i] = np.array((x, y)) 
                ## pos[i] = -np.array((y, -x)) * 1e-6 ## used for single stream
                logger_info(pos[i])
                weight[i] = np.ones_like(raw[i])
                weight[i][np.where(raw[i] == 2 ** 32 - 1)] = 0

                # d = msg['iterable'][0]
                # dct = {'x': d['position'][0],
                #        'y': d['position'][1],
                #        'diff': d['data']}
                logger.info('######## Repackaged data from frame nr. : %d' % i)
            except:
                break
        t1 = time.perf_counter()
        self.loadtottime += t1 - t0
        logger.info('#### Time spent in load = %f, accumulated time = %f' % ((t1 - t0), self.loadtottime))
        logger.info(headerline('', 'c', '#'))
        logger.info(headerline('Leaving LiveScan().load()', 'c', '#'))
        logger.info(headerline('', 'c', '#') + '\n')

        return raw, pos, weight

    def get_LS_data(self):
        return self.loadtottime
