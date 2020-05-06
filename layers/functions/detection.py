import torch
import torch.nn.functional as F
from ..box_utils import decode, jaccard, distance, diou, index2d
from utils import timer

from data import cfg, mask_type

import numpy as np


class Detect(object):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations, as the predicted masks.
    """
    # TODO: Refactor this whole class away. It needs to go.

    def __init__(self, num_classes, bkg_label, top_k, conf_thresh, nms_thresh):
        self.num_classes = num_classes
        self.background_label = bkg_label
        self.top_k = top_k
        # Parameters used in nms.
        self.nms_thresh = nms_thresh
        if nms_thresh <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.conf_thresh = conf_thresh
        
        self.use_cross_class_nms = False
        self.use_fast_nms = False
        self.use_cluster_nms = False
        self.use_cluster_diounms = False
        self.use_spm_nms = False
        self.use_spm_dist_nms = False
        self.use_spm_dist_weighted_nms = False		
		
    def __call__(self, predictions, net):
        """
        Args:
             loc_data: (tensor) Loc preds from loc layers
                Shape: [batch, num_priors, 4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch, num_priors, num_classes]
            mask_data: (tensor) Mask preds from mask layers
                Shape: [batch, num_priors, mask_dim]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors, 4]
            proto_data: (tensor) If using mask_type.lincomb, the prototype masks
                Shape: [batch, mask_h, mask_w, mask_dim]
        
        Returns:
            output of shape (batch_size, top_k, 1 + 1 + 4 + mask_dim)
            These outputs are in the order: class idx, confidence, bbox coords, and mask.

            Note that the outputs are sorted only if cross_class_nms is False
        """

        loc_data   = predictions['loc']
        conf_data  = predictions['conf']
        mask_data  = predictions['mask']
        prior_data = predictions['priors']

        proto_data = predictions['proto'] if 'proto' in predictions else None
        inst_data  = predictions['inst']  if 'inst'  in predictions else None

        out = []

        with timer.env('Detect'):
            batch_size = loc_data.size(0)
            num_priors = prior_data.size(0)

            conf_preds = conf_data.view(batch_size, num_priors, self.num_classes).transpose(2, 1).contiguous()

            for batch_idx in range(batch_size):
                decoded_boxes = decode(loc_data[batch_idx], prior_data)
                result = self.detect(batch_idx, conf_preds, decoded_boxes, mask_data, inst_data)

                if result is not None and proto_data is not None:
                    result['proto'] = proto_data[batch_idx]

                out.append({'detection': result, 'net': net})
        
        return out


    def detect(self, batch_idx, conf_preds, decoded_boxes, mask_data, inst_data):
        """ Perform nms for only the max scoring class that isn't background (class 0) """
        cur_scores = conf_preds[batch_idx, 1:, :]
        conf_scores, _ = torch.max(cur_scores, dim=0)

        keep = (conf_scores > self.conf_thresh)
        scores = cur_scores[:, keep]
        boxes = decoded_boxes[keep, :]
        masks = mask_data[batch_idx, keep, :]

        if inst_data is not None:
            inst = inst_data[batch_idx, keep, :]
    
        if scores.size(1) == 0:
            return None
        
        if self.use_cross_class_nms:
            if self.use_fast_nms:
                boxes, masks, classes, scores = self.cc_fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_cluster_nms:
                boxes, masks, classes, scores = self.cc_cluster_nms(boxes, masks, scores, self.nms_thresh, self.top_k)	
            if self.use_cluster_diounms:
                boxes, masks, classes, scores = self.cc_cluster_diounms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_spm_nms:
                boxes, masks, classes, scores = self.cc_cluster_SPM_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_spm_dist_nms:
                boxes, masks, classes, scores = self.cc_cluster_SPM_dist_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_spm_dist_weighted_nms:
                boxes, masks, classes, scores = self.cc_cluster_SPM_dist_weighted_nms(boxes, masks, scores, self.nms_thresh, self.top_k)						

        else:
            if self.use_fast_nms:
                boxes, masks, classes, scores = self.fast_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_cluster_nms:
                boxes, masks, classes, scores = self.cluster_nms(boxes, masks, scores, self.nms_thresh, self.top_k)	
            if self.use_cluster_diounms:
                boxes, masks, classes, scores = self.cluster_diounms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_spm_nms:
                boxes, masks, classes, scores = self.cluster_SPM_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_spm_dist_nms:
                boxes, masks, classes, scores = self.cluster_SPM_dist_nms(boxes, masks, scores, self.nms_thresh, self.top_k)
            if self.use_spm_dist_weighted_nms:
                boxes, masks, classes, scores = self.cluster_SPM_dist_weighted_nms(boxes, masks, scores, self.nms_thresh, self.top_k)

        return {'box': boxes, 'mask': masks, 'class': classes, 'score': scores}


    def cc_fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
        maxA,_=torch.max(iou, dim=0)
        idx_out = idx[maxA <= iou_threshold]
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def cc_cluster_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_=torch.max(A, dim=0)
            E = (maxA<=iou_threshold).float().unsqueeze(1).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        idx_out = idx[maxA <= iou_threshold]
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def cc_cluster_diounms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        iou = diou(boxes_idx, boxes_idx).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_=torch.max(A, dim=0)
            E = (maxA<=iou_threshold).float().unsqueeze(1).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        idx_out = idx[maxA <= iou_threshold]
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]
		
    def cc_cluster_SPM_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        scores = scores[idx]
        boxes = boxes_idx
        masks = masks[idx]
        classes = classes[idx]
        iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_=torch.max(A, dim=0)
            E = (maxA<=iou_threshold).float().unsqueeze(1).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        scores = torch.prod(torch.exp(-B**2/0.2),0)*scores
        idx_out = scores > 0.01
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]
		
    def cc_cluster_SPM_dist_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        scores = scores[idx]
        boxes = boxes_idx
        masks = masks[idx]
        classes = classes[idx]
        iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_=torch.max(A, dim=0)
            E = (maxA<=iou_threshold).float().unsqueeze(1).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        D=distance(boxes, boxes)
        X = (B>=0).float()
        scores = torch.prod(torch.min(torch.exp(-B**2/0.2)+D*((B>0).float()),X),0)*scores
        idx_out = scores > 0.01

        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]

    def cc_cluster_SPM_dist_weighted_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200):
        # Collapse all the classes into 1 
        scores, classes = scores.max(dim=0)
        _, idx = scores.sort(0, descending=True)
        idx = idx[:top_k]
        boxes_idx = boxes[idx]
        scores = scores[idx]
        boxes = boxes_idx
        masks = masks[idx]
        classes = classes[idx]
        n = len(scores)
        iou = jaccard(boxes_idx, boxes_idx).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_=torch.max(A, dim=0)
            E = (maxA<=iou_threshold).float().unsqueeze(1).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break	
        D=distance(boxes, boxes)
        X = (B>=0).float()
        scores = torch.prod(torch.min(torch.exp(-B**2/0.2)+D*((B>0).float()),X),0)*scores
        idx_out = scores > 0.01
        weights = (B*(B>0.8).float() + torch.eye(n).cuda()) * (scores.reshape((1,n)))
        xx1 = boxes[:,0].expand(n,n)
        yy1 = boxes[:,1].expand(n,n)
        xx2 = boxes[:,2].expand(n,n)
        yy2 = boxes[:,3].expand(n,n)

        weightsum=weights.sum(dim=1)
        xx1 = (xx1*weights).sum(dim=1)/(weightsum)
        yy1 = (yy1*weights).sum(dim=1)/(weightsum)
        xx2 = (xx2*weights).sum(dim=1)/(weightsum)
        yy2 = (yy2*weights).sum(dim=1)/(weightsum)
        boxes = torch.stack([xx1, yy1, xx2, yy2], 1)
        return boxes[idx_out], masks[idx_out], classes[idx_out], scores[idx_out]		

    def fast_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)

        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        iou_max, _ = iou.max(dim=1)

        # Now just filter out the ones higher than the threshold
        keep = (iou_max <= iou_threshold)
        # We should also only keep detections over the confidence threshold, but at the cost of
        # maxing out your detection count for every image, you can just not do that. Because we
        # have such a minimal amount of computation per detection (matrix mulitplication only),
        # this increase doesn't affect us much (+0.2 mAP for 34 -> 33 fps), so we leave it out.
        # However, when you implement this in your method, you should do this second threshold.
        if second_threshold:
            keep *= (scores > self.conf_thresh)
        keep *= (scores > 0.01)
        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]

        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def cluster_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_ = A.max(dim=1)
            E = (maxA <= iou_threshold).float().unsqueeze(2).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        keep = (maxA <= iou_threshold)
        keep *= (scores > 0.01)
        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
		
        return boxes, masks, classes, scores

    def cluster_diounms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = diou(boxes, boxes).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_ = A.max(dim=1)
            E = (maxA <= iou_threshold).float().unsqueeze(2).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        keep = (maxA <= iou_threshold) * (scores > 0.01)
        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]
        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
		
        return boxes, masks, classes, scores

    def cluster_SPM_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_ = A.max(dim=1)
            E = (maxA <= iou_threshold).float().unsqueeze(2).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        scores = torch.prod(torch.exp(-B**2/0.2),1)*scores
        keep = (scores > 0.01)

        #print('keep',torch.sum(keep))
        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
		
        return boxes, masks, classes, scores

    def cluster_SPM_dist_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_ = A.max(dim=1)
            E = (maxA <= iou_threshold).float().unsqueeze(2).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        D=distance(boxes, boxes)
        X = (B>=0).float()
        scores = torch.prod(torch.min(torch.exp(-B**2/0.2)+D*((B>0).float()),X),1)*scores
        keep = (scores > 0.01)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]
		
        return boxes, masks, classes, scores

    def cluster_SPM_dist_weighted_nms(self, boxes, masks, scores, iou_threshold:float=0.5, top_k:int=200, second_threshold:bool=False):
        scores, idx = scores.sort(1, descending=True)
        idx = idx[:, :top_k].contiguous()
        scores = scores[:, :top_k]
    
        num_classes, num_dets = idx.size()

        boxes = boxes[idx.view(-1), :].view(num_classes, num_dets, 4)
        masks = masks[idx.view(-1), :].view(num_classes, num_dets, -1)

        iou = jaccard(boxes, boxes).triu_(diagonal=1)
        B = iou
        for i in range(200):
            A=B
            maxA,_ = A.max(dim=1)
            E = (maxA <= iou_threshold).float().unsqueeze(2).expand_as(A)
            B=iou.mul(E)
            if A.equal(B)==True:
                break
        D=distance(boxes, boxes)
        X = (B>=0).float()
        scores = torch.prod(torch.min(torch.exp(-B**2/0.2)+D*((B>0).float()),X),1)*scores
        keep = (scores > 0.01)
		
        E = keep.float().unsqueeze(2).expand_as(A)
        B=iou.mul(E)
        _,n = scores.size()
        weights = (B*(B>0.8).float() + torch.eye(n).cuda().expand(80,n,n)) * (scores.unsqueeze(2).expand(80,n,n))
        xx1 = boxes[:,:,0].unsqueeze(1).expand(80,n,n)
        yy1 = boxes[:,:,1].unsqueeze(1).expand(80,n,n)
        xx2 = boxes[:,:,2].unsqueeze(1).expand(80,n,n)
        yy2 = boxes[:,:,3].unsqueeze(1).expand(80,n,n)

        weightsum=weights.sum(dim=2)
        xx1 = (xx1*weights).sum(dim=2)/(weightsum)
        yy1 = (yy1*weights).sum(dim=2)/(weightsum)
        xx2 = (xx2*weights).sum(dim=2)/(weightsum)
        yy2 = (yy2*weights).sum(dim=2)/(weightsum)
        boxes = torch.stack([xx1, yy1, xx2, yy2], 2)

        # Assign each kept detection to its corresponding class
        classes = torch.arange(num_classes, device=boxes.device)[:, None].expand_as(keep)
        classes = classes[keep]
        boxes = boxes[keep]
        masks = masks[keep]
        scores = scores[keep]

        # Only keep the top cfg.max_num_detections highest scores across all classes
        scores, idx = scores.sort(0, descending=True)
        idx = idx[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx]
        boxes = boxes[idx]
        masks = masks[idx]

        return boxes, masks, classes, scores

    def traditional_nms_yolact(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []

        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size

        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            conf_mask = cls_scores > conf_thresh
            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            cls_scores = cls_scores[conf_mask]
            idx = idx[conf_mask]

            if cls_scores.size(0) == 0:
                continue
            
            preds = torch.cat([boxes[conf_mask], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()

            idx_lst.append(idx[keep])
            cls_lst.append(keep * 0 + _cls)
            scr_lst.append(cls_scores[keep])
        
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]

        idx = idx[idx2]
        classes = classes[idx2]

        # Undo the multiplication above
        return boxes[idx] / cfg.max_size, masks[idx], classes, scores

    def traditional_nms_ours(self, boxes, masks, scores, iou_threshold=0.5, conf_thresh=0.05):
        import pyximport
        pyximport.install(setup_args={"include_dirs":np.get_include()}, reload_support=True)

        from utils.cython_nms import nms as cnms

        num_classes = scores.size(0)

        idx_lst = []
        cls_lst = []
        scr_lst = []
        box_lst = []
        mask_lst = []
        # Multiplying by max_size is necessary because of how cnms computes its area and intersections
        boxes = boxes * cfg.max_size
        for _cls in range(num_classes):
            cls_scores = scores[_cls, :]
            _, id = cls_scores.sort(0, descending=True)
            id = id[:200].contiguous()
            cls_scores = cls_scores[id]

            idx = torch.arange(cls_scores.size(0), device=boxes.device)

            if cls_scores.size(0) == 0:
                continue
            preds = torch.cat([boxes[id], cls_scores[:, None]], dim=1).cpu().numpy()
            keep = cnms(preds, iou_threshold)
            keep = torch.Tensor(keep, device=boxes.device).long()
            m = (cls_scores[keep] > 0.01)
            idx_lst.append(idx[keep][m])
            cls_lst.append(keep[m] * 0 + _cls)
            scr_lst.append(cls_scores[keep][m])
            box_lst.append(boxes[id][keep][m])
            mask_lst.append(masks[id][keep][m])
        idx     = torch.cat(idx_lst, dim=0)
        classes = torch.cat(cls_lst, dim=0)
        scores  = torch.cat(scr_lst, dim=0)
        boxes  = torch.cat(box_lst, dim=0)
        masks  = torch.cat(mask_lst, dim=0)

        scores, idx2 = scores.sort(0, descending=True)
        idx2 = idx2[:cfg.max_num_detections]
        scores = scores[:cfg.max_num_detections]
        classes = classes[idx2]
        # Undo the multiplication above
        return boxes[idx2] / cfg.max_size, masks[idx2], classes, scores