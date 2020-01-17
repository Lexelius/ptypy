# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""

import numpy as np
import time
from pycuda import gpuarray

from .. import utils as u
from ..utils.verbose import logger, log
from ..utils import parallel
from . import BaseEngine, register, DM_serial, DM
from ..accelerate import py_cuda as gpu
from ..accelerate.py_cuda.kernels import FourierUpdateKernel, AuxiliaryWaveKernel, PoUpdateKernel, PositionCorrectionKernel
from ..accelerate.array_based import address_manglers

MPI = parallel.size > 1
MPI = True

__all__ = ['DM_pycuda']

serialize_array_access = DM_serial.serialize_array_access
gaussian_kernel = DM_serial.gaussian_kernel


@register()
class DM_pycuda(DM_serial.DM_serial):

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine.
        """

        super(DM_pycuda, self).__init__(ptycho_parent, pars)

        self.context, self.queue = gpu.get_context()
        # allocator for READ only buffers
        # self.const_allocator = cl.tools.ImmediateAllocator(queue, cl.mem_flags.READ_ONLY)
        ## gaussian filter
        # dummy kernel
        if not self.p.obj_smooth_std:
            gauss_kernel = gaussian_kernel(1, 1).astype(np.float32)
        else:
            gauss_kernel = gaussian_kernel(self.p.obj_smooth_std, self.p.obj_smooth_std).astype(np.float32)

        self.gauss_kernel_gpu = gpuarray.to_gpu(gauss_kernel)

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        super(DM_pycuda, self).engine_initialize()

        self.error = []

        self.ob_cfact_gpu = {}
        self.pr_cfact_gpu = {}

    def _setup_kernels(self):
        """
        Setup kernels, one for each scan. Derive scans from ptycho class
        """
        # get the scans
        for label, scan in self.ptycho.model.scans.items():

            kern = u.Param()
            self.kernels[label] = kern

            # TODO: needs to be adapted for broad bandwidth
            geo = scan.geometries[0]

            # Get info to shape buffer arrays
            # TODO: make this part of the engine rather than scan
            fpc = self.ptycho.frames_per_block

            # TODO : make this more foolproof
            try:
                nmodes = scan.p.coherence.num_probe_modes * \
                         scan.p.coherence.num_object_modes
            except:
                nmodes = 1

            # create buffer arrays
            ash = (fpc * nmodes,) + tuple(geo.shape)
            aux = np.zeros(ash, dtype=np.complex64)
            kern.aux = gpuarray.to_gpu(aux)

            # setup kernels, one for each SCAN.
            kern.FUK = FourierUpdateKernel(aux, nmodes, queue_thread=self.queue)
            kern.FUK.allocate()

            kern.POK = PoUpdateKernel(queue_thread=self.queue)
            kern.POK.allocate()

            kern.AWK = AuxiliaryWaveKernel(queue_thread=self.queue)
            kern.AWK.allocate()

            from ptypy.accelerate.py_cuda.fft import FFT
            kern.FW = FFT(aux, self.queue,
                          pre_fft=geo.propagator.pre_fft,
                          post_fft=geo.propagator.post_fft,
                          inplace=True,
                          symmetric=True).ft
            kern.BW = FFT(aux, self.queue,
                          pre_fft=geo.propagator.pre_ifft,
                          post_fft=geo.propagator.post_ifft,
                          inplace=True,
                          symmetric=True).ift

            if self.do_position_refinement:
                addr_mangler = address_manglers.RandomIntMangle(int(self.p.position_refinement.amplitude // geo.resolution[0]),
                                                                self.p.position_refinement.start,
                                                                self.p.position_refinement.stop,
                                                                max_bound=int(self.p.position_refinement.max_shift // geo.resolution[0]),
                                                                randomseed=0)
                logger.warning("amplitude is %s " % (self.p.position_refinement.amplitude // geo.resolution[0]))
                logger.warning("max bound is %s " % (self.p.position_refinement.max_shift // geo.resolution[0]))

                kern.PCK = PositionCorrectionKernel(aux, nmodes, queue_thread=self.queue)
                kern.PCK.allocate()
                kern.PCK.address_mangler = addr_mangler
            #self.queue.synchronize()

    def engine_prepare(self):

        super(DM_pycuda, self).engine_prepare()

        ## The following should be restricted to new data

        # recursive copy to gpu
        for _cname, c in self.ptycho.containers.items():
            for _sname, s in c.S.items():
                # convert data here
                if s.data.dtype.name == 'bool':
                    data = s.data.astype(np.float32)
                else:
                    data = s.data
                s.gpu = gpuarray.to_gpu(data)

        for prep in self.diff_info.values():
            
            #prep.addr2 = np.ascontiguousarray(np.transpose(prep.addr, (2, 3, 0, 1)))
            prep.addr = gpuarray.to_gpu(prep.addr)
            #prep.addr2 = gpuarray.to_gpu(prep.addr2)
            prep.mag = gpuarray.to_gpu(prep.mag)
            prep.ma_sum = gpuarray.to_gpu(prep.ma_sum)
            prep.err_fourier = gpuarray.to_gpu(prep.err_fourier)
        self.dummy_error = np.zeros(prep.err_fourier.shape, dtype=np.float32) # np.zeros_like returns 'object' data type

    def engine_iterate(self, num=1):
        """
        Compute one iteration.
        """

        for it in range(num):
            error = {}
            for dID in self.di.S.keys():
                t1 = time.time()

                prep = self.diff_info[dID]
                # find probe, object in exit ID in dependence of dID
                pID, oID, eID = prep.poe_IDs

                # references for kernels
                kern = self.kernels[prep.label]
                FUK = kern.FUK
                AWK = kern.AWK

                pbound = self.pbound_scan[prep.label]
                aux = kern.aux
                FW = kern.FW
                BW = kern.BW

                # get addresses
                addr = prep.addr
                mag = prep.mag
                ma_sum = prep.ma_sum
                err_fourier = prep.err_fourier

                # local references
                ma = self.ma.S[dID].gpu
                ob = self.ob.S[oID].gpu
                pr = self.pr.S[pID].gpu
                ex = self.ex.S[eID].gpu

                queue = self.queue

                t1 = time.time()
                AWK.build_aux(aux, addr, ob, pr, ex, alpha=self.p.alpha)
                # queue.synchronize()

                self.benchmark.A_Build_aux += time.time() - t1

                # FFT
                t1 = time.time()
                FW(aux, aux)

                # queue.synchronize()
                self.benchmark.B_Prop += time.time() - t1

                #  Deviation from measured data
                t1 = time.time()
                FUK.fourier_error(aux, addr, mag, ma, ma_sum)
                FUK.error_reduce(addr, err_fourier)
                FUK.fmag_all_update(aux, addr, mag, ma, err_fourier, pbound)
                # queue.synchronize()
                self.benchmark.C_Fourier_update += time.time() - t1
                # iFFT
                t1 = time.time()
                BW(aux, aux)

                # print("The context is: %s" % self.context)
                # queue.synchronize()
                # print("Here")
                self.benchmark.D_iProp += time.time() - t1

                # apply changes #2
                t1 = time.time()
                AWK.build_exit(aux, addr, ob, pr, ex)
                # queue.synchronize()
                self.benchmark.E_Build_exit += time.time() - t1

                # err_phot = np.zeros_like(err_fourier)
                # err_exit = np.zeros_like(err_fourier)
                # err_err = np.zeros_like(err_fourier)
                # errs = np.array(list(zip(err_err, err_phot, err_exit)))
                errs = np.array(list(zip(self.dummy_error, self.dummy_error, self.dummy_error)))
                error = dict(zip(prep.view_IDs, errs))

                err_fourier_cpu = np.array(err_fourier.get())
                errs = np.ascontiguousarray(np.vstack([err_fourier_cpu, self.dummy_error, self.dummy_error]).T)
                error.update(zip(prep.view_IDs, errs))

                self.benchmark.calls_fourier += 1

            parallel.barrier()

            sync = (self.curiter % 1 == 0)
            self.overlap_update(MPI=MPI)

            parallel.barrier()
            if self.do_position_refinement and (self.curiter):
                do_update_pos = (self.p.position_refinement.stop > self.curiter >= self.p.position_refinement.start)
                do_update_pos &= (self.curiter % self.p.position_refinement.interval) == 0

                # Update positions
                if do_update_pos:
                    """
                    Iterates through all positions and refines them by a given algorithm. 
                    """
                    log(3, "----------- START POS REF -------------")
                    for dID in self.di.S.keys():

                        prep = self.diff_info[dID]
                        pID, oID, eID = prep.poe_IDs
                        ma = self.ma.S[dID].gpu
                        ob = self.ob.S[oID].gpu
                        pr = self.pr.S[pID].gpu
                        kern = self.kernels[prep.label]
                        aux = kern.aux
                        addr = prep.addr
                        original_addr = prep.original_addr
                        mag = prep.mag
                        ma_sum = prep.ma_sum
                        err_fourier = prep.err_fourier

                        PCK = kern.PCK
                        FW = kern.FW

                        error_state = np.zeros(err_fourier.shape, dtype=np.float32)
                        error_state[:] = err_fourier.get()
                        log(4, 'Position refinement trial: iteration %s' % (self.curiter))
                        for i in range(self.p.position_refinement.nshifts):
                            mangled_addr = PCK.address_mangler.mangle_address(addr.get(), original_addr, self.curiter)
                            mangled_addr_gpu = gpuarray.to_gpu(mangled_addr)
                            PCK.build_aux(aux, mangled_addr_gpu, ob, pr)
                            FW(aux, aux)
                            PCK.fourier_error(aux, mangled_addr_gpu, mag, ma, ma_sum)
                            PCK.error_reduce(mangled_addr_gpu, err_fourier)
                            PCK.update_addr_and_error_state(addr, error_state, mangled_addr, err_fourier.get())
                        prep.err_fourier.set(error_state)
                        # prep.addr = addr




            self.curiter += 1
            queue.synchronize()

        for name, s in self.ob.S.items():
            s.data[:] = s.gpu.get()
        for name, s in self.pr.S.items():
            s.data[:] = s.gpu.get()

        # costly but needed to sync back with
        # for name, s in self.ex.S.items():
        #     s.data[:] = s.gpu.get()

        #self.queue.synchronize()

        self.error = error
        return error

    ## object update
    def object_update(self, MPI=False):
        t1 = time.time()
        queue = self.queue
        queue.synchronize()
        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            """
            if self.p.obj_smooth_std is not None:
                logger.info('Smoothing object, cfact is %.2f' % cfact)
                t2 = time.time()
                self.prg.gaussian_filter(queue, (info[3],info[4]), None, obj_gpu.data, self.gauss_kernel_gpu.data)
                queue.synchronize()
                obj_gpu *= cfact
                print 'gauss: '  + str(time.time()-t2)
            else:
                obj_gpu *= cfact
            """
            cfact = self.ob_cfact[oID]
            ob.gpu *= cfact
            # obn.gpu[:] = cfact
            obn.gpu.fill(cfact)
            queue.synchronize()

        # storage for-loop
        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for loop
            ev = POK.ob_update(prep.addr,
                               self.ob.S[oID].gpu,
                               self.ob_nrm.S[oID].gpu,
                               self.pr.S[pID].gpu,
                               self.ex.S[eID].gpu)
            queue.synchronize()

        for oID, ob in self.ob.storages.items():
            obn = self.ob_nrm.S[oID]
            # MPI test
            if MPI:
                ob.data[:] = ob.gpu.get()
                obn.data[:] = obn.gpu.get()
                queue.synchronize()
                parallel.allreduce(ob.data)
                parallel.allreduce(obn.data)
                ob.data /= obn.data

                self.clip_object(ob)
                ob.gpu.set(ob.data)
            else:
                ob.gpu /= obn.gpu

            queue.synchronize()

        # print 'object update: ' + str(time.time()-t1)
        self.benchmark.object_update += time.time() - t1
        self.benchmark.calls_object += 1

    ## probe update
    def probe_update(self, MPI=False):
        t1 = time.time()
        queue = self.queue

        # storage for-loop
        change = 0
        cfact = self.p.probe_inertia
        for pID, pr in self.pr.storages.items():
            prn = self.pr_nrm.S[pID]
            cfact = self.pr_cfact[pID]
            pr.gpu *= cfact
            prn.gpu.fill(cfact)

        for dID in self.di.S.keys():
            prep = self.diff_info[dID]

            POK = self.kernels[prep.label].POK
            # find probe, object in exit ID in dependence of dID
            pID, oID, eID = prep.poe_IDs

            # scan for-loop
            ev = POK.pr_update(prep.addr,
                               self.pr.S[pID].gpu,
                               self.pr_nrm.S[pID].gpu,
                               self.ob.S[oID].gpu,
                               self.ex.S[eID].gpu)
            queue.synchronize()

        for pID, pr in self.pr.storages.items():

            buf = self.pr_buf.S[pID]
            prn = self.pr_nrm.S[pID]

            # MPI test
            if MPI:
                # if False:
                pr.data[:] = pr.gpu.get()
                prn.data[:] = prn.gpu.get()
                queue.synchronize()
                parallel.allreduce(pr.data)
                parallel.allreduce(prn.data)
                pr.data /= prn.data

                self.support_constraint(pr)

                pr.gpu.set(pr.data)
            else:
                pr.gpu /= prn.gpu
                # ca. 0.3 ms
                # self.pr.S[pID].gpu = probe_gpu
                pr.data[:] = pr.gpu.get()

            ## this should be done on GPU

            queue.synchronize()
            change += u.norm2(pr.data - buf.data) / u.norm2(pr.data)
            buf.data[:] = pr.data
            if MPI:
                change = parallel.allreduce(change) / parallel.size

        # print 'probe update: ' + str(time.time()-t1)
        self.benchmark.probe_update += time.time() - t1
        self.benchmark.calls_probe += 1

        return np.sqrt(change)

    def engine_finalize(self):
        """
        try deleting ever helper contianer
        """
        super(DM_pycuda, self).engine_finalize()
        #self.queue.synchronize()
        self.context.detach()

        # delete local references to container buffer copies