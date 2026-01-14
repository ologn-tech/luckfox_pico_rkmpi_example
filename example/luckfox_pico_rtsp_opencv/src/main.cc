/*****************************************************************************
* | Author      :   Luckfox team
* | Function    :
* | Info        :
*
*----------------
* | This version:   V2.0
* | Date        :   2024-08-26
* | Info        :   Basic version
*
******************************************************************************/

#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <getopt.h>
#include <linux/videodev2.h>
#include <sys/fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <pthread.h>
#include <signal.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/poll.h>
#include <time.h>
#include <unistd.h>
#include <vector>


#include <rk_aiq_user_api2_camgroup.h>
#include <rk_aiq_user_api2_imgproc.h>
#include <rk_aiq_user_api2_sysctl.h>
#include "rk_smart_ir_api.h"
#include "rtsp_demo.h"
#include "luckfox_mpi.h"
#include "sample_comm.h"

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#ifndef WIDTH
#define DISP_WIDTH  1920
#else
#define DISP_WIDTH  WIDTH
#endif

#ifndef HEIGHT
#define DISP_HEIGHT 1080
#else
#define DISP_HEIGHT HEIGHT
#endif

static rk_aiq_sys_ctx_t *aiq_ctx = NULL;
static rk_aiq_static_info_t aiq_static_info = {0};
static rk_smart_ir_ctx_t *smartIr_ctx = NULL;
static void smartIr_cb(rk_smart_ir_result_t result)
{
    if (result.status == RK_SMART_IR_STATUS_NIGHT) {
        if (result.is_status_change) {
            printf("SMART_IR: switch to Night\n");
            // 1) switch isp night params
            rk_aiq_uapi2_sysctl_switch_scene(aiq_ctx, "normal", "night");
            // 2) ir-cutter off
            // TODO: user should define ir-cutter control func here
        }
        if (result.is_fill_change) {
            // 3) manual/auto ir-led, set result.fill_value
            // TODO: user should define led control func here
        }
    } else if (result.status == RK_SMART_IR_STATUS_DAY && result.is_status_change) {
        printf("SMART_IR: switch to Day\n");
        // 1) ir-cutter on
        // TODO: user should define ir-cutter control func here
        // 2) ir-led off
        // TODO: user should define led control func here
        // 3) switch isp day params
        rk_aiq_uapi2_sysctl_switch_scene(aiq_ctx, "normal", "day");
    }
}
static void smartIr_start()
{
    smartIr_ctx = rk_smart_ir_init(aiq_ctx);
    /* NOTE:
     * The API `rk_smart_ir_iniCfg` reads configuration from an INI file located at the configured path.
     * If the API `rk_smart_ir_setAttr` is called after `rk_smart_ir_iniCfg`, the configuration read from
     * the INI file might be overwritten by the new settings.
     */
    rk_smart_ir_iniCfg(smartIr_ctx, "tmp/smart_ir.ini");

    rk_smart_ir_attr_t attr;
    //memset(&attr, 0, sizeof(attr));
    rk_smart_ir_getAttr(smartIr_ctx, &attr);
    attr.init_status = RK_SMART_IR_STATUS_DAY;
    attr.switch_mode = RK_SMART_IR_SWITCH_MODE_AUTO;
    attr.light_mode = RK_SMART_IR_LIGHT_MODE_MANUAL;
    attr.light_type = RK_SMART_IR_LIGHT_TYPE_IR;
    attr.light_value = 100;
    attr.params.d2n_envL_th = 0.04f;
    attr.params.n2d_envL_th = 0.20f;
    attr.params.rggain_base = 1.00f;
    attr.params.bggain_base = 1.00f;
    attr.params.awbgain_rad = 0.10f;
    attr.params.awbgain_dis = 0.20f;
    attr.params.switch_cnts_th = 50;
    rk_smart_ir_setAttr(smartIr_ctx, &attr);
    rk_smart_ir_runCb(smartIr_ctx, false, smartIr_cb);
}

static void smartIr_stop()
{
    rk_smart_ir_deInit(smartIr_ctx);
    smartIr_ctx = NULL;
}

void print_usage(const char *program_name) {
	printf("Usage: %s [OPTIONS]\n", program_name);
	printf("Options:\n");
	printf("  -m, --mode MODE    Set day/night mode (day or night, default: day)\n");
	printf("  -h, --help         Show this help message\n");
}

int main(int argc, char *argv[]) {
  	system("RkLunch-stop.sh");
	RK_S32 s32Ret = 0;

	int width    = DISP_WIDTH;
	int height   = DISP_HEIGHT;

	// Parse command line arguments
	const char *mode = "day";  // default to day mode
	int opt;
	struct option long_options[] = {
		{"mode", required_argument, 0, 'm'},
		{"help", no_argument, 0, 'h'},
		{0, 0, 0, 0}
	};

	while ((opt = getopt_long(argc, argv, "m:h", long_options, NULL)) != -1) {
		switch (opt) {
			case 'm':
				if (strcmp(optarg, "day") == 0 || strcmp(optarg, "night") == 0) {
					mode = optarg;
				} else {
					fprintf(stderr, "Error: Invalid mode '%s'. Use 'day' or 'night'.\n", optarg);
					print_usage(argv[0]);
					return -1;
				}
				break;
			case 'h':
				print_usage(argv[0]);
				return 0;
			default:
				print_usage(argv[0]);
				return -1;
		}
	}

	printf("Using mode: %s\n", mode);

	char fps_text[16];
	float fps = 0;
	memset(fps_text,0,16);
	//h264_frame
	VENC_STREAM_S stFrame;
	stFrame.pstPack = (VENC_PACK_S *)malloc(sizeof(VENC_PACK_S));
	RK_U64 H264_PTS = 0;
	RK_U32 H264_TimeRef = 0;
	VIDEO_FRAME_INFO_S stViFrame;

	// Create Pool
	MB_POOL_CONFIG_S PoolCfg;
	memset(&PoolCfg, 0, sizeof(MB_POOL_CONFIG_S));
	PoolCfg.u64MBSize = width * height * 3 ;
	PoolCfg.u32MBCnt = 1;
	PoolCfg.enAllocType = MB_ALLOC_TYPE_DMA;
	//PoolCfg.bPreAlloc = RK_FALSE;
	MB_POOL src_Pool = RK_MPI_MB_CreatePool(&PoolCfg);
	printf("Create Pool success !\n");

	// Get MB from Pool
	MB_BLK src_Blk = RK_MPI_MB_GetMB(src_Pool, width * height * 3, RK_TRUE);

	// Build h264_frame
	VIDEO_FRAME_INFO_S h264_frame;
	h264_frame.stVFrame.u32Width = width;
	h264_frame.stVFrame.u32Height = height;
	h264_frame.stVFrame.u32VirWidth = width;
	h264_frame.stVFrame.u32VirHeight = height;
	h264_frame.stVFrame.enPixelFormat =  RK_FMT_RGB888;
	h264_frame.stVFrame.u32FrameFlag = 160;
	h264_frame.stVFrame.pMbBlk = src_Blk;
	unsigned char *data = (unsigned char *)RK_MPI_MB_Handle2VirAddr(src_Blk);
	cv::Mat frame(cv::Size(width,height),CV_8UC3,data);

	// rkaiq init
	rk_aiq_uapi2_sysctl_enumStaticMetas(0, &aiq_static_info);

	aiq_ctx = rk_aiq_uapi2_sysctl_init(aiq_static_info.sensor_info.sensor_name, "/etc/iqfiles", NULL, NULL);

	if (rk_aiq_uapi2_sysctl_prepare(aiq_ctx, 0, 0, RK_AIQ_WORKING_MODE_NORMAL)) {
		RK_LOGE("rk_aiq_uapi_sysctl_prepare fail!");
		return -1;
	}

	if (rk_aiq_uapi2_sysctl_start(aiq_ctx)) {
		RK_LOGE("rk_aiq_uapi_sysctl_start fail!");
                return -1;
        }

	rk_aiq_uapi2_sysctl_switch_scene(aiq_ctx, "normal", mode);

    smartIr_start();
	// rkmpi init
	if (RK_MPI_SYS_Init() != RK_SUCCESS) {
		RK_LOGE("rk mpi sys init fail!");
		return -1;
	}

	// rtsp init
	rtsp_demo_handle g_rtsplive = NULL;
	rtsp_session_handle g_rtsp_session;
	g_rtsplive = create_rtsp_demo(554);
	g_rtsp_session = rtsp_new_session(g_rtsplive, "/live/0");
	rtsp_set_video(g_rtsp_session, RTSP_CODEC_ID_VIDEO_H265, NULL, 0);
	rtsp_sync_video_ts(g_rtsp_session, rtsp_get_reltime(), rtsp_get_ntptime());

	// vi init
	vi_dev_init();
	vi_chn_init(0, width, height);

	// venc init
	RK_CODEC_ID_E enCodecType = RK_VIDEO_ID_HEVC;
	venc_init(0, width, height, enCodecType);

	printf("init success\n");

	while(1) {
		// get vi frame
		h264_frame.stVFrame.u32TimeRef = H264_TimeRef++;
		h264_frame.stVFrame.u64PTS = TEST_COMM_GetNowUs();
		s32Ret = RK_MPI_VI_GetChnFrame(0, 0, &stViFrame, -1);
		if(s32Ret == RK_SUCCESS)
		{
			void *vi_data = RK_MPI_MB_Handle2VirAddr(stViFrame.stVFrame.pMbBlk);

			cv::Mat yuv420sp(height + height / 2, width, CV_8UC1, vi_data);
			cv::Mat bgr(height, width, CV_8UC3, data);
			cv::cvtColor(yuv420sp, bgr, cv::COLOR_YUV420sp2BGR);
			cv::resize(bgr, frame, cv::Size(width ,height), 0, 0, cv::INTER_LINEAR);

			sprintf(fps_text,"fps = %.2f",fps);
                        cv::putText(frame, fps_text, cv::Point(40, 40),
                                	cv::FONT_HERSHEY_SIMPLEX, 1,
                                	cv::Scalar(0, 255, 0), 2);
                }
		memcpy(data, frame.data, width * height * 3);
		// encode H264
		RK_MPI_VENC_SendFrame(0,  &h264_frame ,-1);
		// rtsp
		s32Ret = RK_MPI_VENC_GetStream(0, &stFrame, -1);
		if(s32Ret == RK_SUCCESS) {
			if(g_rtsplive && g_rtsp_session) {
				//printf("len = %d PTS = %d \n",stFrame.pstPack->u32Len, stFrame.pstPack->u64PTS);
				void *pData = RK_MPI_MB_Handle2VirAddr(stFrame.pstPack->pMbBlk);
				rtsp_tx_video(g_rtsp_session, (uint8_t *)pData, stFrame.pstPack->u32Len,
							  stFrame.pstPack->u64PTS);
				rtsp_do_event(g_rtsplive);
			}
			RK_U64 nowUs = TEST_COMM_GetNowUs();
			fps = (float) 1000000 / (float)(nowUs - h264_frame.stVFrame.u64PTS);
		}
		// release frame
		s32Ret = RK_MPI_VI_ReleaseChnFrame(0, 0, &stViFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VI_ReleaseChnFrame fail %x", s32Ret);
		}
		s32Ret = RK_MPI_VENC_ReleaseStream(0, &stFrame);
		if (s32Ret != RK_SUCCESS) {
			RK_LOGE("RK_MPI_VENC_ReleaseStream fail %x", s32Ret);
		}
	}

	// Destory MB
	RK_MPI_MB_ReleaseMB(src_Blk);
	// Destory Pool
	RK_MPI_MB_DestroyPool(src_Pool);

	RK_MPI_VI_DisableChn(0, 0);
	RK_MPI_VI_DisableDev(0);

	SAMPLE_COMM_ISP_Stop(0);

	RK_MPI_VENC_StopRecvFrame(0);
	RK_MPI_VENC_DestroyChn(0);

	free(stFrame.pstPack);

	if (g_rtsplive)
		rtsp_del_demo(g_rtsplive);

	RK_MPI_SYS_Exit();
    smartIr_stop();
    if (aiq_ctx) {
        rk_aiq_uapi2_sysctl_stop(aiq_ctx, false);
        rk_aiq_uapi2_sysctl_deinit(aiq_ctx);
        aiq_ctx = NULL;
    }

	return 0;
}
