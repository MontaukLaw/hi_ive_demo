#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/ioctl.h>
#include "sample_comm.h"
#include "hi_ive.h"
#include "hi_comm_ive.h"
#include "mpi_ive.h"
#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>

using namespace std;

#define LOG(ERROR) cout << "ERROR:"

static pthread_t vpss_get_frame_thread_id;
static HI_BOOL sys_running = HI_TRUE;
#define HI_SUCC HI_SUCCESS
#define HI_FAIL HI_FAILURE

using namespace cv;

typedef struct tagIPC_IMAGE
{
    HI_U64 u64PhyAddr;
    HI_U64 u64VirAddr;
    HI_U32 u32Width;
    HI_U32 u32Height;
} IPC_IMAGE;

HI_S32 yuvFrame2rgb(VIDEO_FRAME_INFO_S *srcFrame, IPC_IMAGE *dstImage)
{
    IVE_HANDLE hIveHandle;
    IVE_SRC_IMAGE_S pstSrc;
    IVE_DST_IMAGE_S pstDst;
    IVE_CSC_CTRL_S stCscCtrl;
    HI_S32 s32Ret = 0;
    stCscCtrl.enMode = IVE_CSC_MODE_PIC_BT709_YUV2RGB; // IVE_CSC_MODE_VIDEO_BT601_YUV2RGB;
    pstSrc.enType = IVE_IMAGE_TYPE_YUV420SP;
    pstSrc.au64VirAddr[0] = srcFrame->stVFrame.u64VirAddr[0];
    pstSrc.au64VirAddr[1] = srcFrame->stVFrame.u64VirAddr[1];
    pstSrc.au64VirAddr[2] = srcFrame->stVFrame.u64VirAddr[2];

    pstSrc.au64PhyAddr[0] = srcFrame->stVFrame.u64PhyAddr[0];
    pstSrc.au64PhyAddr[1] = srcFrame->stVFrame.u64PhyAddr[1];
    pstSrc.au64PhyAddr[2] = srcFrame->stVFrame.u64PhyAddr[2];

    pstSrc.au32Stride[0] = srcFrame->stVFrame.u32Stride[0];
    pstSrc.au32Stride[1] = srcFrame->stVFrame.u32Stride[1];
    pstSrc.au32Stride[2] = srcFrame->stVFrame.u32Stride[2];

    pstSrc.u32Width = srcFrame->stVFrame.u32Width;
    pstSrc.u32Height = srcFrame->stVFrame.u32Height;

    pstDst.enType = IVE_IMAGE_TYPE_U8C3_PACKAGE;
    pstDst.u32Width = pstSrc.u32Width;
    pstDst.u32Height = pstSrc.u32Height;
    pstDst.au32Stride[0] = pstSrc.au32Stride[0];
    pstDst.au32Stride[1] = 0;
    pstDst.au32Stride[2] = 0;
    s32Ret = HI_MPI_SYS_MmzAlloc_Cached(&pstDst.au64PhyAddr[0], (void **)&pstDst.au64VirAddr[0], "User", HI_NULL, pstDst.u32Height * pstDst.au32Stride[0] * 3);
    if (HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        LOG(ERROR) << "HI_MPI_SYS_MmzAlloc_Cached Failed with 0x" << hex << s32Ret << endl;
        return s32Ret;
    }
    s32Ret = HI_MPI_SYS_MmzFlushCache(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0], pstDst.u32Height * pstDst.au32Stride[0] * 3);
    if (HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        LOG(ERROR) << "HI_MPI_SYS_MmzFlushCache Failed with 0x" << hex << s32Ret << endl;
        return s32Ret;
    }
    memset((void *)pstDst.au64VirAddr[0], 0, pstDst.u32Height * pstDst.au32Stride[0] * 3);
    HI_BOOL bInstant = HI_TRUE;
    s32Ret = HI_MPI_IVE_CSC(&hIveHandle, &pstSrc, &pstDst, &stCscCtrl, bInstant);
    if (HI_SUCCESS != s32Ret)
    {
        HI_MPI_SYS_MmzFree(pstDst.au64PhyAddr[0], (void *)pstDst.au64VirAddr[0]);
        LOG(ERROR) << "HI_MPI_IVE_CSC Failed with 0x" << hex << s32Ret << endl;
        return s32Ret;
    }
    if (HI_TRUE == bInstant)
    {
        HI_BOOL bFinish = HI_TRUE;
        HI_BOOL bBlock = HI_TRUE;
        s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        while (HI_ERR_IVE_QUERY_TIMEOUT == s32Ret)
        {
            usleep(100);
            s32Ret = HI_MPI_IVE_Query(hIveHandle, &bFinish, bBlock);
        }
    }
    dstImage->u64PhyAddr = pstDst.au64PhyAddr[0];
    dstImage->u64VirAddr = pstDst.au64VirAddr[0];
    dstImage->u32Width = pstDst.u32Width;
    dstImage->u32Height = pstDst.u32Height;
    return HI_SUCC;
}

HI_S32 frame2Mat(VIDEO_FRAME_INFO_S *srcFrame, Mat &dstMat)
{
    HI_U32 w = srcFrame->stVFrame.u32Width;
    HI_U32 h = srcFrame->stVFrame.u32Height;
    int bufLen = w * h * 3;
    HI_U8 *srcRGB = NULL;
    IPC_IMAGE dstImage;
    if (yuvFrame2rgb(srcFrame, &dstImage) != HI_SUCC)
    {
        LOG(ERROR) << "yuvFrame2rgb Err ." << endl;
        return HI_FAIL;
    }
    srcRGB = (HI_U8 *)dstImage.u64VirAddr;
    dstMat.create(h, w, CV_8UC3);
    memcpy(dstMat.data, srcRGB, bufLen * sizeof(HI_U8));
    HI_MPI_SYS_MmzFree(dstImage.u64PhyAddr, (void *)&(dstImage.u64VirAddr));
    return HI_SUCC;
}

HI_VOID *vpss_get_frame_thread(HI_VOID *params)
{
    HI_S32 s32Ret = HI_SUCCESS;

    VPSS_GRP VpssGrp = 0;
    VPSS_CHN VpssChn = VPSS_CHN0;
    VIDEO_FRAME_INFO_S stVFrame;

    HI_S32 s32MilliSec = 100;
    Mat mat;

    long counter = 0;

    // HI_S32 HI_MPI_VPSS_GetChnFrame(VPSS_GRP VpssGrp, VPSS_CHN VpssChn,VIDEO_FRAME_INFO_S *pstVideoFrame, HI_S32 s32MilliSec);
    while (sys_running)
    {
        s32Ret = HI_MPI_VPSS_GetChnFrame(VpssGrp, VpssChn, &stVFrame, s32MilliSec);
        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("HI_MPI_VPSS_GetChnFrame failed with %#x!\n", s32Ret);
            continue;
        }

        SAMPLE_PRT("HI_MPI_VPSS_GetChnFrame success!\n");
        frame2Mat(&stVFrame, mat);

        // 写入
        if (counter == 10)
        {
            imwrite("test.jpg", mat);
        }

        counter++;

        s32Ret = HI_MPI_VPSS_ReleaseChnFrame(VpssGrp, VpssChn, &stVFrame);
        if (HI_SUCCESS != s32Ret)
        {
            SAMPLE_PRT("HI_MPI_VPSS_ReleaseChnFrame failed with %#x!\n", s32Ret);
            continue;
        }
    }

    return NULL;
}

HI_VOID SAMPLE_VIO_MsgInit(HI_VOID)
{
}

HI_VOID SAMPLE_VIO_MsgExit(HI_VOID)
{
}

void SAMPLE_VIO_HandleSig(HI_S32 signo)
{
    signal(SIGINT, SIG_IGN);
    signal(SIGTERM, SIG_IGN);

    if (SIGINT == signo || SIGTERM == signo)
    {
        sys_running = HI_FALSE;
        SAMPLE_COMM_VENC_StopGetStream();
        SAMPLE_COMM_All_ISP_Stop();
        SAMPLE_COMM_VO_HdmiStop();
        SAMPLE_COMM_SYS_Exit();
        SAMPLE_PRT("\033[0;31mprogram termination abnormally!\033[0;39m\n");
    }
    exit(-1);
}

HI_S32 SAMPLE_VIO_ViOnlineVpssOnlineRoute(HI_U32 u32VoIntfType)
{
    HI_S32 s32Ret = HI_SUCCESS;

    HI_S32 s32ViCnt = 1;
    VI_DEV ViDev = 1;
    VI_PIPE ViPipe = 0;
    VI_CHN ViChn = 0;
    HI_S32 s32WorkSnsId = 0;
    SAMPLE_VI_CONFIG_S stViConfig;

    SIZE_S stSize;
    VB_CONFIG_S stVbConf;
    PIC_SIZE_E enPicSize;
    HI_U32 u32BlkSize;

    VO_CHN VoChn = 0;
    SAMPLE_VO_CONFIG_S stVoConfig;

    WDR_MODE_E enWDRMode = WDR_MODE_NONE;
    DYNAMIC_RANGE_E enDynamicRange = DYNAMIC_RANGE_SDR8;
    PIXEL_FORMAT_E enPixFormat = PIXEL_FORMAT_YVU_SEMIPLANAR_420;
    VIDEO_FORMAT_E enVideoFormat = VIDEO_FORMAT_LINEAR;
    COMPRESS_MODE_E enCompressMode = COMPRESS_MODE_NONE;
    VI_VPSS_MODE_E enMastPipeMode = VI_ONLINE_VPSS_ONLINE;

    VPSS_GRP VpssGrp = 0;
    VPSS_GRP_ATTR_S stVpssGrpAttr;
    VPSS_CHN VpssChn = VPSS_CHN0;
    HI_BOOL abChnEnable[VPSS_MAX_PHY_CHN_NUM] = {HI_FALSE};
    VPSS_CHN_ATTR_S astVpssChnAttr[VPSS_MAX_PHY_CHN_NUM];

    VENC_CHN VencChn[1] = {0};
    PAYLOAD_TYPE_E enType = PT_H265;
    SAMPLE_RC_E enRcMode = SAMPLE_RC_CBR;
    HI_U32 u32Profile = 0;
    HI_BOOL bRcnRefShareBuf = HI_FALSE;
    VENC_GOP_ATTR_S stGopAttr;

    /*config vi*/
    SAMPLE_COMM_VI_GetSensorInfo(&stViConfig);

    SAMPLE_PRT("ViDev:%d, ViPipe:%d, ViChn:%d, s32WorkSnsId:%d", ViDev, ViPipe, ViChn, s32WorkSnsId);

    stViConfig.s32WorkingViNum = s32ViCnt;
    stViConfig.as32WorkingViId[0] = 0;
    stViConfig.astViInfo[s32WorkSnsId].stSnsInfo.MipiDev = ViDev;
    stViConfig.astViInfo[s32WorkSnsId].stSnsInfo.s32BusId = 1;
    stViConfig.astViInfo[s32WorkSnsId].stDevInfo.ViDev = ViDev;
    stViConfig.astViInfo[s32WorkSnsId].stDevInfo.enWDRMode = enWDRMode;
    stViConfig.astViInfo[s32WorkSnsId].stPipeInfo.enMastPipeMode = enMastPipeMode;
    stViConfig.astViInfo[s32WorkSnsId].stPipeInfo.aPipe[0] = ViPipe;
    stViConfig.astViInfo[s32WorkSnsId].stPipeInfo.aPipe[1] = -1;
    stViConfig.astViInfo[s32WorkSnsId].stPipeInfo.aPipe[2] = -1;
    stViConfig.astViInfo[s32WorkSnsId].stPipeInfo.aPipe[3] = -1;
    stViConfig.astViInfo[s32WorkSnsId].stChnInfo.ViChn = ViChn;
    stViConfig.astViInfo[s32WorkSnsId].stChnInfo.enPixFormat = enPixFormat;
    stViConfig.astViInfo[s32WorkSnsId].stChnInfo.enDynamicRange = enDynamicRange;
    stViConfig.astViInfo[s32WorkSnsId].stChnInfo.enVideoFormat = enVideoFormat;
    stViConfig.astViInfo[s32WorkSnsId].stChnInfo.enCompressMode = enCompressMode;

    /*get picture size*/
    s32Ret = SAMPLE_COMM_VI_GetSizeBySensor(stViConfig.astViInfo[s32WorkSnsId].stSnsInfo.enSnsType, &enPicSize);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("get picture size by sensor failed!\n");
        return s32Ret;
    }

    s32Ret = SAMPLE_COMM_SYS_GetPicSize(enPicSize, &stSize);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("get picture size failed!\n");
        return s32Ret;
    }

    /*config vb*/
    memset_s(&stVbConf, sizeof(VB_CONFIG_S), 0, sizeof(VB_CONFIG_S));
    stVbConf.u32MaxPoolCnt = 2;

    u32BlkSize = COMMON_GetPicBufferSize(stSize.u32Width, stSize.u32Height, SAMPLE_PIXEL_FORMAT, DATA_BITWIDTH_8, COMPRESS_MODE_SEG, DEFAULT_ALIGN);
    stVbConf.astCommPool[0].u64BlkSize = u32BlkSize;
    stVbConf.astCommPool[0].u32BlkCnt = 10;

    u32BlkSize = VI_GetRawBufferSize(stSize.u32Width, stSize.u32Height, PIXEL_FORMAT_RGB_BAYER_16BPP, COMPRESS_MODE_NONE, DEFAULT_ALIGN);
    stVbConf.astCommPool[1].u64BlkSize = u32BlkSize;
    stVbConf.astCommPool[1].u32BlkCnt = 4;

    s32Ret = SAMPLE_COMM_SYS_Init(&stVbConf);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("system init failed with %d!\n", s32Ret);
        return s32Ret;
    }

    /*start vi*/
    s32Ret = SAMPLE_COMM_VI_StartVi(&stViConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("start vi failed.s32Ret:0x%x !\n", s32Ret);
        goto EXIT;
    }

    /*config vpss*/
    memset_s(&stVpssGrpAttr, sizeof(VPSS_GRP_ATTR_S), 0, sizeof(VPSS_GRP_ATTR_S));
    stVpssGrpAttr.stFrameRate.s32SrcFrameRate = -1;
    stVpssGrpAttr.stFrameRate.s32DstFrameRate = -1;
    stVpssGrpAttr.enDynamicRange = DYNAMIC_RANGE_SDR8;
    stVpssGrpAttr.enPixelFormat = enPixFormat;
    stVpssGrpAttr.u32MaxW = stSize.u32Width;
    stVpssGrpAttr.u32MaxH = stSize.u32Height;
    stVpssGrpAttr.bNrEn = HI_TRUE;
    stVpssGrpAttr.stNrAttr.enCompressMode = COMPRESS_MODE_FRAME;
    stVpssGrpAttr.stNrAttr.enNrMotionMode = NR_MOTION_MODE_NORMAL;

    astVpssChnAttr[VpssChn].u32Width = stSize.u32Width;
    astVpssChnAttr[VpssChn].u32Height = stSize.u32Height;
    astVpssChnAttr[VpssChn].enChnMode = VPSS_CHN_MODE_USER;
    astVpssChnAttr[VpssChn].enCompressMode = enCompressMode;
    astVpssChnAttr[VpssChn].enDynamicRange = enDynamicRange;
    astVpssChnAttr[VpssChn].enVideoFormat = enVideoFormat;
    astVpssChnAttr[VpssChn].enPixelFormat = enPixFormat;
    astVpssChnAttr[VpssChn].stFrameRate.s32SrcFrameRate = 30;
    astVpssChnAttr[VpssChn].stFrameRate.s32DstFrameRate = 30;
    astVpssChnAttr[VpssChn].u32Depth = 1;
    astVpssChnAttr[VpssChn].bMirror = HI_FALSE;
    astVpssChnAttr[VpssChn].bFlip = HI_FALSE;
    astVpssChnAttr[VpssChn].stAspectRatio.enMode = ASPECT_RATIO_NONE;

    /*start vpss*/
    abChnEnable[0] = HI_TRUE;
    s32Ret = SAMPLE_COMM_VPSS_Start(VpssGrp, abChnEnable, &stVpssGrpAttr, astVpssChnAttr);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("start vpss group failed. s32Ret: 0x%x !\n", s32Ret);
        goto EXIT1;
    }

#if 0
    /*config venc */
    stGopAttr.enGopMode = VENC_GOPMODE_SMARTP;
    stGopAttr.stSmartP.s32BgQpDelta = 7;
    stGopAttr.stSmartP.s32ViQpDelta = 2;
    stGopAttr.stSmartP.u32BgInterval = 1200;
    s32Ret = SAMPLE_COMM_VENC_Start(VencChn[0], enType, enPicSize, enRcMode, u32Profile, bRcnRefShareBuf, &stGopAttr);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("start venc failed. s32Ret: 0x%x !\n", s32Ret);
        goto EXIT2;
    }

    s32Ret = SAMPLE_COMM_VPSS_Bind_VENC(VpssGrp, VpssChn, VencChn[0]);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Venc bind Vpss failed. s32Ret: 0x%x !n", s32Ret);
        goto EXIT3;
    }
#endif

    /*config vo*/
    SAMPLE_COMM_VO_GetDefConfig(&stVoConfig);
    stVoConfig.enDstDynamicRange = enDynamicRange;
    if (1 == u32VoIntfType)
    {
        stVoConfig.enVoIntfType = VO_INTF_BT1120;
        stVoConfig.enIntfSync = VO_OUTPUT_1080P25;
    }
    else
    {
        stVoConfig.enVoIntfType = VO_INTF_HDMI;
    }
    stVoConfig.enPicSize = enPicSize;

    /*start vo*/
    s32Ret = SAMPLE_COMM_VO_StartVO(&stVoConfig);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("start vo failed. s32Ret: 0x%x !\n", s32Ret);
        goto EXIT4;
    }

    /*vpss bind vo*/
    s32Ret = SAMPLE_COMM_VPSS_Bind_VO(VpssGrp, VpssChn, stVoConfig.VoDev, VoChn);
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("vo bind vpss failed. s32Ret: 0x%x !\n", s32Ret);
        goto EXIT5;
    }

#if 0
    s32Ret = SAMPLE_COMM_VENC_StartGetStream(VencChn, sizeof(VencChn) / sizeof(VENC_CHN));
    if (HI_SUCCESS != s32Ret)
    {
        SAMPLE_PRT("Get venc stream failed!\n");
        goto EXIT6;
    }
#endif

    pthread_create(&vpss_get_frame_thread_id, 0, vpss_get_frame_thread, NULL);

    PAUSE();

    SAMPLE_COMM_VENC_StopGetStream();

EXIT6:
    SAMPLE_COMM_VPSS_UnBind_VO(VpssGrp, VpssChn, stVoConfig.VoDev, VoChn);
EXIT5:
    SAMPLE_COMM_VO_StopVO(&stVoConfig);
EXIT4:
    SAMPLE_COMM_VPSS_UnBind_VENC(VpssGrp, VpssChn, VencChn[0]);
EXIT3:
    SAMPLE_COMM_VENC_Stop(VencChn[0]);
EXIT2:
    SAMPLE_COMM_VPSS_Stop(VpssGrp, abChnEnable);
EXIT1:
    SAMPLE_COMM_VI_StopVi(&stViConfig);
EXIT:
    SAMPLE_COMM_SYS_Exit();
    return s32Ret;
}

int main(void)
{

    HI_U32 u32VoIntfType = 0;
    HI_U32 u32ChipId;

    HI_MPI_SYS_GetChipId(&u32ChipId);

    if (HI3516C_V500 == u32ChipId)
    {
        u32VoIntfType = 1;
    }
    else
    {
        u32VoIntfType = 0;
    }

    SAMPLE_VIO_ViOnlineVpssOnlineRoute(0);
}
