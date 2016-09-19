# -*- coding: utf-8 -*-
"""
Difference Map reconstruction engine.
Independent-Probe flavour.

This file is part of the PTYPY package.

    :copyright: Copyright 2014 by the PTYPY team, see AUTHORS.
    :license: GPLv2, see LICENSE for details.
"""
import numpy as np
import time
from .. import utils as u
from ..utils.verbose import logger
from ..utils import parallel
from engine_utils import basic_fourier_update
from . import DM

__all__ = ['DMIP']

DEFAULT = u.Param(
    alpha=1,                       # Difference map parameter
    probe_update_start=2,          # Number of iterations before probe update starts
    update_object_first=True,      # If True update object before probe
    overlap_converge_factor=0.05,  # Threshold for interruption of the inner overlap loop
    overlap_max_iterations=10,     # Maximum of iterations for the overlap constraint inner loop
    probe_metric=1,                # Weight of the current probe estimate in the update, similar to probe_inertia but
                                   # crucial here.
    object_inertia=1e-4,           # Weight of the current object in the update, formally DM_smooth_amplitude
    fourier_relax_factor=0.05,     # If rms error of model vs diffraction data is smaller than this fraction,
                                   # Fourier constraint is met
    obj_smooth_std=None,           # Gaussian smoothing (pixel) of the current object prior to update
    clip_object=None,              # None or tuple(min,max) of desired limits of the object modulus,
                                   # currently in under common in documentation
)

    
class DMIP(DM):
    
    DEFAULT = DEFAULT

    def __init__(self, ptycho_parent, pars=None):
        """
        Difference map reconstruction engine - independent probe flavour.
        """
        if pars is None:
            pars = DEFAULT.copy()
            
        super(DMIP, self).__init__(ptycho_parent, pars)

        # Instance attributes
        self.idprobe = None

    def engine_initialize(self):
        """
        Prepare for reconstruction.
        """
        self.error = []

        # Generate container copies
        self.ob_buf = self.ob.copy(self.ob.ID + '_alt', fill=0.)
        self.ob_nrm = self.ob.copy(self.ob.ID + '_nrm', fill=0.)
        self.ob_viewcover = self.ob.copy(self.ob.ID + '_vcover', fill=0.)

        self.pr_buf = self.pr.copy(self.pr.ID + '_alt', fill=0.)
        self.pr_nrm = self.pr.copy(self.pr.ID + '_nrm', fill=0.)

        self.idprobe = self.exit.copy(self.exit.ID + 'IP', fill=0.)

    def engine_prepare(self):
        """
        Last minute initialization. Everything that needs to be recalculated when new data arrives.
        """

        self.pbound = {}
        for name, s in self.di.S.iteritems():
            self.pbound[name] = .25 * self.p.fourier_relax_factor**2 * s.pbound_stub
        
        # Fill object with coverage of views
        for name, s in self.ob_viewcover.S.iteritems():
            s.fill(s.get_view_coverage())

    def engine_iterate(self, num=1):
        """
        Compute `num` iterations.
        """
        to = 0.
        tf = 0.
        for it in range(num):
            t1 = time.time() 
            
            # Fourier update  
            error_dct = self.fourier_update()

            t2 = time.time()
            tf += t2 - t1
            
            # Overlap update
            self.overlap_update()
            
            t3 = time.time()
            to += t3 - t2
            
        logger.info('Time spent in Fourier update: %.2f' % tf)
        logger.info('Time spent in Overlap update: %.2f' % to)
        error = parallel.gather_dict(error_dct)
        return error

    def engine_finalize(self):
        """
        Try deleting ever helper container.
        """
        containers = [
            self.ob_buf,
            self.ob_nrm,
            self.ob_viewcover,
            self.pr_buf,
            self.pr_nrm]

        for c in containers:
            logger.debug('Attempt to remove container %s' % c.ID)
            del self.ptycho.containers[c.ID]
        #    IDM.used.remove(c.ID)
        
        del self.ob_buf
        del self.ob_nrm 
        del self.ob_viewcover 
        del self.pr_buf
        del self.pr_nrm

        del containers

    def fourier_update(self):
        """
        DM Fourier constraint update (including DM step)
        """
        error_dct = {}
        for name, di_view in self.di.V.iteritems():
            if not di_view.active:
                continue
            pbound = self.pbound[di_view.storage.ID]
            error_dct[name] = basic_fourier_update(di_view, pbound=pbound, alpha=self.p.alpha)
        return error_dct

    def overlap_update(self):
        """
        DM overlap constraint update.
        """
        # Condition to update probe
        do_update_probe = (self.p.probe_update_start <= self.curiter)
         
        for inner in range(self.p.overlap_max_iterations):
            pre_str = 'Iteration (Overlap) #%02d:  ' % inner
            
            # Update object first
            if self.p.update_object_first or (inner > 0):
                # Update object
                logger.debug(pre_str + '----- object update -----')
                self.object_update()
                               
            # Exit if probe should not be updated yet
            if not do_update_probe:
                break
            
            # Update probe
            logger.debug(pre_str + '----- probe update -----')
            change = self.probe_update()
            logger.debug(pre_str + 'change in probe is %.3f' % change)
            
            # Recenter the probe
            self.center_probe()
            
            # Stop iteration if probe change is small
            if change < self.p.overlap_converge_factor:
                break

    def center_probe(self):
        if self.p.probe_center_tol is not None:
            for name, s in self.pr.S.iteritems():
                c1 = u.mass_center(u.abs2(s.data).sum(0))
                c2 = np.asarray(s.shape[-2:]) // 2       # fft convention should however use geometry instead
                if u.norm(c1 - c2) < self.p.probe_center_tol:
                    break
                # SC: possible BUG here, wrong input parameter
                s.data[:] = u.shift_zoom(s.data, (1.,) * 3, (0, c1[0], c1[1]), (0, c2[0], c2[1]))
                logger.info('Probe recentered from %s to %s' % (str(tuple(c1)), str(tuple(c2))))
                
    def object_update(self):
        """
        DM object update.
        """
        ob = self.ob
        ob_nrm = self.ob_nrm
        
        # Fill container
        if not parallel.master:
            ob.fill(0.0)
            ob_nrm.fill(0.)
        else:
            for name, s in self.ob.S.iteritems():
                # in original code:
                # DM_smooth_amplitude = (p.DM_smooth_amplitude * max_power * p.num_probes * Ndata) / np.prod(asize)
                # using the number of views here, but don't know if that is good.
                # cfact = self.p.object_inertia * len(s.views)
                cfact = self.p.object_inertia * (self.ob_viewcover.S[name].data + 1.)
                
                if self.p.obj_smooth_std is not None:
                    logger.info('Smoothing object, average cfact is %.2f + %.2fj' %
                                (np.mean(cfact).real, np.mean(cfact).imag))
                    smooth_mfs = [0, self.p.obj_smooth_std, self.p.obj_smooth_std]
                    s.data[:] = cfact * u.c_gf(s.data, smooth_mfs)
                else:
                    s.data[:] = s.data * cfact
                    
                ob_nrm.S[name].fill(cfact)
        
        # DM update per node
        for name, pod in self.pods.iteritems():
            if not pod.active:
                continue
            pod.object += pod.probe.conj() * pod.exit * pod.object_weight
            ob_nrm[pod.ob_view] += u.cabs2(pod.probe) * pod.object_weight
        
        # Distribute result with MPI
        for name, s in self.ob.S.iteritems():
            # Get the np arrays
            nrm = ob_nrm.S[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm
                
            # Clip object
            if self.p.clip_object is not None:
                clip_min, clip_max = self.p.clip_object
                ampl_obj = np.abs(s.data)
                phase_obj = np.exp(1j * np.angle(s.data))
                too_high = (ampl_obj > clip_max)
                too_low = (ampl_obj < clip_min)
                s.data[too_high] = clip_max * phase_obj[too_high]
                s.data[too_low] = clip_min * phase_obj[too_low]
                
    def probe_update(self):
        """
        DM probe update.
        """
        pr = self.pr
        pr_nrm = self.pr_nrm
        pr_buf = self.pr_buf
        
        # Fill container
        # "cfact" fill
        # BE: was this asymmetric in original code only because of the number of MPI nodes ?
        if parallel.master:
            for name, s in pr.S.iteritems():
                # instead of Npts_scan, the number of views should be considered
                # please note that a call to s.views maybe slow for many views in the probe.
                cfact = self.p.probe_inertia * len(s.views) / s.data.shape[0]
                s.data[:] = cfact * s.data
                pr_nrm.S[name].fill(cfact)
        else:
            pr.fill(0.0)
            pr_nrm.fill(0.0)

        # DM update per node
        for name, pod in self.pods.iteritems():
            if not pod.active:
                continue
            pod.probe += pod.object.conj() * pod.exit * pod.probe_weight
            pr_nrm[pod.pr_view] += u.cabs2(pod.object) * pod.probe_weight

        change = 0.
        
        # Distribute result with MPI
        for name, s in pr.S.iteritems():
            # MPI reduction of results
            nrm = pr_nrm.S[name].data
            parallel.allreduce(s.data)
            parallel.allreduce(nrm)
            s.data /= nrm
            
            # Apply probe support if requested
            support = self.probe_support.get(name)
            if support is not None: 
                s.data *= self.probe_support[name]

            # Compute relative change in probe
            buf = pr_buf.S[name].data
            change += u.norm2(s.data - buf) / u.norm2(s.data)

            # Fill buffer with new probe
            buf[:] = s.data

        return np.sqrt(change / len(pr.S))
