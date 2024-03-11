
#import <AVFoundation/AVFoundation.h>

extern NSString *const THThumbnailCreatedNotification;

@protocol THCameraControllerDelegate <NSObject>

// 1 error handling
- (void)deviceConfigurationFailedWithError:(NSError *)error;
- (void)mediaCaptureFailedWithError:(NSError *)error;
- (void)assetLibraryWriteFailedWithError:(NSError *)error;
@end

@interface THCameraController : NSObject

@property (weak, nonatomic) id<THCameraControllerDelegate> delegate;
@property (nonatomic, strong, readonly) AVCaptureSession *captureSession;


// 2 setup, start and stop AVCapture sessions
- (BOOL)setupSession:(NSError **)error;
- (void)startSession;
- (void)stopSession;

// 3 front and back camera switching
- (BOOL)switchCameras;
- (BOOL)canSwitchCameras;
@property (nonatomic, readonly) NSUInteger cameraCount;
@property (nonatomic, readonly) BOOL cameraHasTorch; // torch
@property (nonatomic, readonly) BOOL cameraHasFlash; // flashlight
@property (nonatomic, readonly) BOOL cameraSupportsTapToFocus; // focus
@property (nonatomic, readonly) BOOL cameraSupportsTapToExpose;// exposure
@property (nonatomic) AVCaptureTorchMode torchMode; // torch mode
@property (nonatomic) AVCaptureFlashMode flashMode; // flash moode

// 4 focus, exposure and reset
- (void)focusAtPoint:(CGPoint)point;
- (void)exposeAtPoint:(CGPoint)point;
- (void)resetFocusAndExposureModes;

// 5 capture photos & videos

// still images
- (void)captureStillImage;

// video recording
- (void)startRecording;

- (void)stopRecording;

- (BOOL)isRecording;

- (CMTime)recordedDuration;

@end
