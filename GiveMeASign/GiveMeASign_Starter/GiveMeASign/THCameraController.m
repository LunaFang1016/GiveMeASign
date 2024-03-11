
#import "THCameraController.h"
#import <AVFoundation/AVFoundation.h>
#import <AssetsLibrary/AssetsLibrary.h>
#import "NSFileManager+THAdditions.h"

NSString *const THThumbnailCreatedNotification = @"THThumbnailCreated";

@interface THCameraController () <AVCaptureFileOutputRecordingDelegate>

@property (strong, nonatomic) dispatch_queue_t videoQueue;
@property (strong, nonatomic) AVCaptureSession *captureSession;
@property (weak, nonatomic) AVCaptureDeviceInput *activeVideoInput;
@property (strong, nonatomic) AVCaptureStillImageOutput *imageOutput;
@property (strong, nonatomic) AVCaptureMovieFileOutput *movieOutput;
@property (strong, nonatomic) NSURL *outputURL;

@end

@implementation THCameraController

- (BOOL)setupSession:(NSError **)error {

    
    // create an AVCaptureSession (core session)
    self.captureSession = [[AVCaptureSession alloc]init];
    
    /*
     AVCaptureSessionPresetHigh
     AVCaptureSessionPresetMedium
     AVCaptureSessionPresetLow
     AVCaptureSessionPreset640x480
     AVCaptureSessionPreset1280x720
     AVCaptureSessionPresetPhoto
     */
    self.captureSession.sessionPreset = AVCaptureSessionPreset1280x720;
    
    AVCaptureDevice *videoDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeVideo];
    
    AVCaptureDeviceInput *videoInput = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:error];
    
    if (videoInput)
    {
        // canAddInpu
        if ([self.captureSession canAddInput:videoInput])
        {
            // add videoInput into captureSession中
            [self.captureSession addInput:videoInput];
            self.activeVideoInput = videoInput;
        }
    }else
    {
        return NO;
    }
    
    // default audio input
    AVCaptureDevice *audioDevice = [AVCaptureDevice defaultDeviceWithMediaType:AVMediaTypeAudio];
    
    // audio input device
    AVCaptureDeviceInput *audioInput = [AVCaptureDeviceInput deviceInputWithDevice:audioDevice error:error];
   
    // check audio input
    if (audioInput) {
        
        // canAddInput
        if ([self.captureSession canAddInput:audioInput])
        {
            // add audioInput into captureSession
            [self.captureSession addInput:audioInput];
        }
    }else
    {
        return NO;
    }

    // TODO: update with AVCapturePhotoOutput
    // AVCaptureStillImageOutput instance
    self.imageOutput = [[AVCaptureStillImageOutput alloc]init];
    
    // photo type settings
    self.imageOutput.outputSettings = @{AVVideoCodecKey:AVVideoCodecJPEG};
    
    // output connection
    if ([self.captureSession canAddOutput:self.imageOutput])
    {
        [self.captureSession addOutput:self.imageOutput];
        
    }
    
    
    // AVCaptureMovieFileOutput instance to output recorded video as .mov files
    self.movieOutput = [[AVCaptureMovieFileOutput alloc]init];
    
    // output connection
    if ([self.captureSession canAddOutput:self.movieOutput])
    {
        [self.captureSession addOutput:self.movieOutput];
    }
    
    
    self.videoQueue = dispatch_queue_create("cmu.e1.GiveMeASign", NULL);
    
    return YES;
}

- (void)startSession {

    if (![self.captureSession isRunning])
    {
        dispatch_async(self.videoQueue, ^{
            [self.captureSession startRunning];
        });
        
    }
}

- (void)stopSession {

    if ([self.captureSession isRunning])
    {
        dispatch_async(self.videoQueue, ^{
            [self.captureSession stopRunning];
        });
    }
    


}

//- (dispatch_queue_t)globalQueue {
//    
//    return dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_DEFAULT, 0);
//}

#pragma mark - Device Configuration

- (AVCaptureDevice *)cameraWithPosition:(AVCaptureDevicePosition)position {
    
    // get available devices
    NSArray *devicess = [AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo];
    
    for (AVCaptureDevice *device in devicess)
    {
        if (device.position == position) {
            return device;
        }
    }
    return nil;
    
    
}

- (AVCaptureDevice *)activeCamera {

    return self.activeVideoInput.device;
}

- (AVCaptureDevice *)inactiveCamera {

       AVCaptureDevice *device = nil;
      if (self.cameraCount > 1)
      {
          if ([self activeCamera].position == AVCaptureDevicePositionBack) {
               device = [self cameraWithPosition:AVCaptureDevicePositionFront];
         }else
         {
             device = [self cameraWithPosition:AVCaptureDevicePositionBack];
         }
     }

    return device;
    

}

- (BOOL)canSwitchCameras {

    
    return self.cameraCount > 1;
}

- (NSUInteger)cameraCount {

     return [[AVCaptureDevice devicesWithMediaType:AVMediaTypeVideo] count];
    
}

- (BOOL)switchCameras {

    if (![self canSwitchCameras])
    {
        return NO;
    }
    
    NSError *error;
    AVCaptureDevice *videoDevice = [self inactiveCamera];
    
    
    AVCaptureDeviceInput *videoInput = [AVCaptureDeviceInput deviceInputWithDevice:videoDevice error:&error];
    
    if (videoInput)
    {
        
        [self.captureSession beginConfiguration];
        
        [self.captureSession removeInput:self.activeVideoInput];
        
        if ([self.captureSession canAddInput:videoInput])
        {
            [self.captureSession addInput:videoInput];
            
            self.activeVideoInput = videoInput;
        }else
        {
            [self.captureSession addInput:self.activeVideoInput];
        }
        
        [self.captureSession commitConfiguration];
    }else
    {
        [self.delegate deviceConfigurationFailedWithError:error];
        return NO;
    }
    
    
    
    return YES;
}

/*
    AVCapture Device
 */


#pragma mark - Focus Methods

- (BOOL)cameraSupportsTapToFocus {
    
    return [[self activeCamera]isFocusPointOfInterestSupported];
}

- (void)focusAtPoint:(CGPoint)point {
    
    AVCaptureDevice *device = [self activeCamera];
    
    if (device.isFocusPointOfInterestSupported && [device isFocusModeSupported:AVCaptureFocusModeAutoFocus]) {
        
        NSError *error;
        if ([device lockForConfiguration:&error]) {
            
            device.focusPointOfInterest = point;
            
            device.focusMode = AVCaptureFocusModeAutoFocus;
            
            [device unlockForConfiguration];
        }else{
            [self.delegate deviceConfigurationFailedWithError:error];
        }
        
    }
    
}

#pragma mark - Exposure Methods

- (BOOL)cameraSupportsTapToExpose {
    
    return [[self activeCamera] isExposurePointOfInterestSupported];
}

static const NSString *THCameraAdjustingExposureContext;

- (void)exposeAtPoint:(CGPoint)point {

    
    AVCaptureDevice *device = [self activeCamera];
    
    AVCaptureExposureMode exposureMode =AVCaptureExposureModeContinuousAutoExposure;
    
    if (device.isExposurePointOfInterestSupported && [device isExposureModeSupported:exposureMode]) {
        
        [device isExposureModeSupported:exposureMode];
        
        NSError *error;
        
        if ([device lockForConfiguration:&error])
        {
            device.exposurePointOfInterest = point;
            device.exposureMode = exposureMode;
            
            if ([device isExposureModeSupported:AVCaptureExposureModeLocked]) {
                
                // support: use kvo to determine the device's adjustingExposure cndition
                [device addObserver:self forKeyPath:@"adjustingExposure" options:NSKeyValueObservingOptionNew context:&THCameraAdjustingExposureContext];
                
            }
            
            // release lock
            [device unlockForConfiguration];
            
        }else
        {
            [self.delegate deviceConfigurationFailedWithError:error];
        }
        
        
    }
    
}

- (void)observeValueForKeyPath:(NSString *)keyPath
                      ofObject:(id)object
                        change:(NSDictionary *)change
                       context:(void *)context {

    // check THCameraAdjustingExposureContext
    if (context == &THCameraAdjustingExposureContext) {
        
        AVCaptureDevice *device = (AVCaptureDevice *)object;
        
        if(!device.isAdjustingExposure && [device isExposureModeSupported:AVCaptureExposureModeLocked])
        {
            [object removeObserver:self forKeyPath:@"adjustingExposure" context:&THCameraAdjustingExposureContext];
            
            dispatch_async(dispatch_get_main_queue(), ^{
                NSError *error;
                if ([device lockForConfiguration:&error]) {
                    
                    device.exposureMode = AVCaptureExposureModeLocked;
                    
                    [device unlockForConfiguration];
                    
                }else
                {
                    [self.delegate deviceConfigurationFailedWithError:error];
                }
            });
            
        }
        
    }else
    {
        [super observeValueForKeyPath:keyPath ofObject:object change:change context:context];
    }
    
    
}

- (void)resetFocusAndExposureModes {

    
    AVCaptureDevice *device = [self activeCamera];
    
    
    
    AVCaptureFocusMode focusMode = AVCaptureFocusModeContinuousAutoFocus;
    
    BOOL canResetFocus = [device isFocusPointOfInterestSupported]&& [device isFocusModeSupported:focusMode];
    
    AVCaptureExposureMode exposureMode = AVCaptureExposureModeContinuousAutoExposure;
    
    BOOL canResetExposure = [device isFocusPointOfInterestSupported] && [device isExposureModeSupported:exposureMode];
    
    CGPoint centPoint = CGPointMake(0.5f, 0.5f);
    
    NSError *error;
    
    if ([device lockForConfiguration:&error]) {
        
        if (canResetFocus) {
            device.focusMode = focusMode;
            device.focusPointOfInterest = centPoint;
        }
        
        if (canResetExposure) {
            device.exposureMode = exposureMode;
            device.exposurePointOfInterest = centPoint;
            
        }
        
        [device unlockForConfiguration];
        
    }else
    {
        [self.delegate deviceConfigurationFailedWithError:error];
    }
    
    
    
    
}



#pragma mark - Flash and Torch Modes

- (BOOL)cameraHasFlash {

    return [[self activeCamera]hasFlash];

}

// TODO: replace deprecated methods
- (AVCaptureFlashMode)flashMode {

    
    return [[self activeCamera]flashMode];
}

- (void)setFlashMode:(AVCaptureFlashMode)flashMode {

    AVCaptureDevice *device = [self activeCamera];
    
    if ([device isFlashModeSupported:flashMode]) {
    
        NSError *error;
        if ([device lockForConfiguration:&error]) {

            device.flashMode = flashMode;
            [device unlockForConfiguration];
            
        }else
        {
            [self.delegate deviceConfigurationFailedWithError:error];
        }
        
    }

}

- (BOOL)cameraHasTorch {

    return [[self activeCamera]hasTorch];
}

- (AVCaptureTorchMode)torchMode {

    return [[self activeCamera]torchMode];
}


- (void)setTorchMode:(AVCaptureTorchMode)torchMode {

    
    AVCaptureDevice *device = [self activeCamera];
    
    if ([device isTorchModeSupported:torchMode]) {
        
        NSError *error;
        if ([device lockForConfiguration:&error]) {
            
            device.torchMode = torchMode;
            [device unlockForConfiguration];
        }else
        {
            [self.delegate deviceConfigurationFailedWithError:error];
        }

    }
    
}


#pragma mark - Image Capture Methods
/*
    AVCaptureStillImageOutput
 */
- (void)captureStillImage {
    
    AVCaptureConnection *connection = [self.imageOutput connectionWithMediaType:AVMediaTypeVideo];
    
    if (connection.isVideoOrientationSupported) {
        
        connection.videoOrientation = [self currentVideoOrientation];
    }
    
    // return NSData info
    id handler = ^(CMSampleBufferRef sampleBuffer,NSError *error)
                {
                    if (sampleBuffer != NULL) {
                        NSData *imageData = [AVCaptureStillImageOutput jpegStillImageNSDataRepresentation:sampleBuffer];
                        UIImage *image = [[UIImage alloc]initWithData:imageData];
                        
                        // success
                        [self writeImageToAssetsLibrary:image];
                    }else
                    {
                        NSLog(@"NULL sampleBuffer:%@",[error localizedDescription]);
                    }
                        
                };
    
    [self.imageOutput captureStillImageAsynchronouslyFromConnection:connection completionHandler:handler];
    
    
    
}

// orientation
- (AVCaptureVideoOrientation)currentVideoOrientation {
    
    AVCaptureVideoOrientation orientation;
    
    switch ([UIDevice currentDevice].orientation) {
        case UIDeviceOrientationPortrait:
            orientation = AVCaptureVideoOrientationPortrait;
            break;
        case UIDeviceOrientationLandscapeRight:
            orientation = AVCaptureVideoOrientationLandscapeLeft;
            break;
        case UIDeviceOrientationPortraitUpsideDown:
            orientation = AVCaptureVideoOrientationPortraitUpsideDown;
            break;
        default:
            orientation = AVCaptureVideoOrientationLandscapeRight;
            break;
    }
    
    return orientation;

    return 0;
}


/*
    Assets Library
    Get access to photo album
 */

- (void)writeImageToAssetsLibrary:(UIImage *)image {

    // ALAssetsLibrary
    ALAssetsLibrary *library = [[ALAssetsLibrary alloc]init];
    
    // param1: CGImageRef
    // param2: orientation --> NSUInteger
    // param3: success/error
    [library writeImageToSavedPhotosAlbum:image.CGImage
                             orientation:(NSUInteger)image.imageOrientation
                         completionBlock:^(NSURL *assetURL, NSError *error) {
                             // send preview image notification
                             if (!error)
                             {
                                 [self postThumbnailNotifification:image];
                             }else
                             {
                                 // error message
                                 id message = [error localizedDescription];
                                 NSLog(@"%@",message);
                             }
                         }];
}

// preview image notification
- (void)postThumbnailNotifification:(UIImage *)image {
    
    dispatch_async(dispatch_get_main_queue(), ^{
        NSNotificationCenter *nc = [NSNotificationCenter defaultCenter];
        [nc postNotificationName:THThumbnailCreatedNotification object:image];
    });
}

#pragma mark - Video Capture Methods

- (BOOL)isRecording {

    return self.movieOutput.isRecording;
}

// start recording
- (void)startRecording {

    if (![self isRecording]) {
        
        AVCaptureConnection * videoConnection = [self.movieOutput connectionWithMediaType:AVMediaTypeVideo];
        
        // videoOrientation
        if([videoConnection isVideoOrientationSupported])
        {
            videoConnection.videoOrientation = [self currentVideoOrientation];
            
        }
        
        if([videoConnection isVideoStabilizationSupported])
        {
            videoConnection.enablesVideoStabilizationWhenAvailable = YES;
        }
        
        
        AVCaptureDevice *device = [self activeCamera];
        
        if (device.isSmoothAutoFocusEnabled) {
            NSError *error;
            if ([device lockForConfiguration:&error]) {
                
                device.smoothAutoFocusEnabled = YES;
                [device unlockForConfiguration];
            }else
            {
                [self.delegate deviceConfigurationFailedWithError:error];
            }
        }
        
        // search for unique path
        self.outputURL = [self uniqueURL];
        
        // first param: recording save path  param2: delegate
        [self.movieOutput startRecordingToOutputFileURL:self.outputURL recordingDelegate:self];
        
    }
    
    
}

- (CMTime)recordedDuration {
    
    return self.movieOutput.recordedDuration;
}


- (NSURL *)uniqueURL {

    NSFileManager *fileManager = [NSFileManager defaultManager];
    
    //temporaryDirectoryWithTemplateString
    NSString *dirPath = [fileManager temporaryDirectoryWithTemplateString:@"givemeasign.XXXXXX"];
    
    if (dirPath) {
        
        NSString *filePath = [dirPath stringByAppendingPathComponent:@"givemeasign_movie.mov"];
        return  [NSURL fileURLWithPath:filePath];
        
    }
    
    return nil;
    
}

- (void)stopRecording {

    if ([self isRecording]) {
        [self.movieOutput stopRecording];
    }
}

#pragma mark - AVCaptureFileOutputRecordingDelegate

- (void)captureOutput:(AVCaptureFileOutput *)captureOutput
didFinishRecordingToOutputFileAtURL:(NSURL *)outputFileURL
      fromConnections:(NSArray *)connections
                error:(NSError *)error {

    if (error) {
        [self.delegate mediaCaptureFailedWithError:error];
    }else
    {
        [self writeVideoToAssetsLibrary:[self.outputURL copy]];
        
    }
    
    self.outputURL = nil;
    

}

// write captured video
// TODO: update methods to higher iOS versions
- (void)writeVideoToAssetsLibrary:(NSURL *)videoURL {
    
    //ALAssetsLibrary instance for recording writing interface
    ALAssetsLibrary *library = [[ALAssetsLibrary alloc]init];
    
    // check if able to write
    if ([library videoAtPathIsCompatibleWithSavedPhotosAlbum:videoURL]) {
        
        ALAssetsLibraryWriteVideoCompletionBlock completionBlock;
        completionBlock = ^(NSURL *assetURL,NSError *error)
        {
            if (error) {
                
                [self.delegate assetLibraryWriteFailedWithError:error];
            }else
            {
                // for preview
                [self generateThumbnailForVideoAtURL:videoURL];
            }
            
        };
        
        [library writeVideoAtPathToSavedPhotosAlbum:videoURL completionBlock:completionBlock];
    }
}

- (void)generateThumbnailForVideoAtURL:(NSURL *)videoURL {

    dispatch_async(self.videoQueue, ^{
        
        AVAsset *asset = [AVAsset assetWithURL:videoURL];
        
        AVAssetImageGenerator *imageGenerator = [AVAssetImageGenerator assetImageGeneratorWithAsset:asset];
        
        // set size by scale
        imageGenerator.maximumSize = CGSizeMake(100.0f, 0.0f);
        
        imageGenerator.appliesPreferredTrackTransform = YES;
        
        CGImageRef imageRef = [imageGenerator copyCGImageAtTime:kCMTimeZero actualTime:NULL error:nil];
        
        UIImage *image = [UIImage imageWithCGImage:imageRef];
        
        CGImageRelease(imageRef);
        
        dispatch_async(dispatch_get_main_queue(), ^{
            
            [self postThumbnailNotifification:image];
            
        });
        
    });
    
}


@end

