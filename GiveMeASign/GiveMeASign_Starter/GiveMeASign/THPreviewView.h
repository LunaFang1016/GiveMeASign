
#import <AVFoundation/AVFoundation.h>

@protocol THPreviewViewDelegate <NSObject>
- (void)tappedToFocusAtPoint:(CGPoint)point; // foucs
- (void)tappedToExposeAtPoint:(CGPoint)point; // exposure
- (void)tappedToResetFocusAndExposure; // focus & exposure reset
@end

@interface THPreviewView : UIView

//session to connect AVCaptureVideoPreviewLayer and to activate AVCaptureSession
@property (strong, nonatomic) AVCaptureSession *session;
@property (weak, nonatomic) id<THPreviewViewDelegate> delegate;

@property (nonatomic) BOOL tapToFocusEnabled; // check focus status
@property (nonatomic) BOOL tapToExposeEnabled; // check exposure status

@end
