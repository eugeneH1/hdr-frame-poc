# HDR video rendering assignment

For this assignment we'd like you to create a POC of rec2100/rec2020 HDR video frame rendering inside the browser.
The code may be tailored specifically to fit the assignment, any examples of useful code reuse for future expansion is great, but not mandatory.

While it has to run inside the browser, there's no specific requirement for how you go about the compositing, generation of video frames, library or tool use. Feel free to use the available native APIs or involve WASM as you see fit. Listing a reason for each such decision is part of the assignment, and that's part of what we may discuss retroactively.

**Duration** Around one or two evenings, if it's not enough time to reach a functional result that's fine. We'd much rather you have an outline we can use to talk about some of the considerations or blockers of the approach than spend too much time getting it running. 

## Requirements
- Repository (github, gitlab, other)
  - Containing your implementation
  - README detailing some of your decisions, tool usage and approach
- Rendering is contained within the browser enviroment (Chrome, any flags allowed), without involving a separate service
- 1280x720 over 10 seconds at 30fps
- Background box, HDR colored
- 3 shapes 
  - 100x100 
  - Each a different color in the rec2100 or rec2020 colorspace
  - Can be any shape, rects are fine
  - Each shape animates between frames
    - Various matrix transforms (translate, rotate, scale, skew)
    - Opacity
- Float array (or similar wrapper construct) frames are returned as a result of the rendering process
- Frames are visually shown during rendering, an SDR estimation is sufficient

## Nice to haves
- Planning of further optimisation
- Description of potential issues for this application in the browser environment
- Raw frames are fed into a videoencoder (either native WebCodecs/VideoEncoder, or a WASM implementation) to produce a true HDR video
- Output frames are shown accurately (as HDR) during the rendering process
- Rendering text

Do note there's no hard requirement for producing a true video at the end of this pipeline, generating the frames of the video is the core assignment.

Don't hesitate to reach out if you have any questions about this assignment! Looking forward to see how you'll approach the problem.
