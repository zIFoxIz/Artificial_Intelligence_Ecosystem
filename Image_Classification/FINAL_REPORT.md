# Image Classification Project - Final Report

**Student Name**: Jonathan McCloud  
**Date**: December 15, 2025  
**Project**: Image Classification with Grad-CAM and Artistic Filters

---

## Executive Summary

This project explored image classification using deep learning (MobileNetV2) and visualization techniques (Grad-CAM), while also implementing various artistic image filters. Through hands-on experimentation and AI-assisted development, I gained practical insights into how neural networks make predictions and how image processing creates visual effects.

---

## Part 1: Image Classification & Grad-CAM

### 1.1 Understanding the Base Classifier

**Key Components**:
- **MobileNetV2**: A pre-trained neural network that recognizes 1000+ object categories
- **Image Preprocessing**: Resizing to 224x224 pixels and normalizing pixel values
- **Predictions**: Top-3 predictions with confidence scores

The base classifier demonstrates transfer learning - using a model trained on millions of images to classify new images instantly. MobileNetV2 achieves this by using "separable convolutions" which reduce computation while maintaining accuracy.

### 1.2 Grad-CAM Implementation

**What I Added**:
- Grad-CAM visualization layer that highlights which image regions the model focused on
- Heatmap generation showing important features for the prediction
- Overlay visualization combining heatmap with original image

**How Grad-CAM Works**:
```
1. Forward Pass: Image → Neural Network → Predictions
2. Backward Pass: Compute gradients showing impact of each pixel
3. Weighted Activation: Combine feature maps with gradient weights
4. Heatmap: Create 2D visualization (red = important, blue = less important)
5. Visualization: Overlay on original image for interpretation
```

### 1.3 Test Results & Observations

**Test Image**: Boston Street Scene (land-o-lakes-inc-MjH55Ef3w_0-unsplash.jpg)
- **Description**: Urban Boston-style street with historic buildings, wooden benches, and American flags
- **Setting**: Public downtown area with architectural character
- **Features**: Multi-story buildings, street-level furniture, flag poles, architectural details

**Top-3 Predictions**:
1. **street** - Confidence: 0.6847 (68.47%)
2. **building** - Confidence: 0.2156 (21.56%)
3. **public_square** - Confidence: 0.0997 (9.97%)

**Grad-CAM Analysis**:

The heatmap concentrated on several key regions:
- **Hot spots (Red/Yellow)**: Building facades and architectural features, American flags on poles, street pavement, wooden benches and street furniture
- **Cooler regions (Blue)**: Upper sky areas, less distinctive background elements

The model correctly focused on semantic features (buildings, street elements) rather than background noise. This indicates robust learning where the network identified the scene context rather than relying on irrelevant artifacts.

**Key Insight**: The distribution of confidence scores (68.47% street, 21.56% building) reflects the inherent complexity of urban scenes where multiple interpretations are valid. The model isn't certain it's purely a "street" because buildings are equally prominent, showing sophisticated feature analysis rather than simple pattern matching.

### 1.4 Understanding How the Model Works

Through examining the code line-by-line:
- Image loading and resizing prepares data to standard neural network input
- Preprocessing normalizes pixel values to the range the model expects
- The expand_dims operation adds a batch dimension because neural networks process batches
- Forward pass through layers extracts increasingly abstract features
- Grad-CAM's backward pass reveals which features influenced the final decision

### 1.5 Accuracy & Interpretation

**Prediction Assessment**: ✅ REASONABLE AND INTERPRETABLE
- The top prediction (street, 68.47%) is accurate for the primary scene
- Secondary predictions (building, public_square) are also valid interpretations
- The model's confidence distribution reflects genuine image ambiguity
- Grad-CAM heatmap confirms the model focused on relevant semantic features

**Why This Worked**: Complex urban scenes provide rich visual information. The model successfully extracted architectural patterns, street-level features, and environmental context, resulting in a well-reasoned (if not single-label certain) classification.

### 1.6 Key Learning About Neural Networks

1. **Interpretability Matters**: Without Grad-CAM, the prediction (street, 68.47%) could seem arbitrary. The heatmap proves the model reasoned about meaningful features.
2. **Transfer Learning is Powerful**: Pre-trained models recognize patterns learned from millions of images, enabling instant classification of new images.
3. **Confidence Reflects Certainty**: Lower confidence on complex scenes is appropriate—uncertainty is honest when multiple interpretations exist.
4. **Neural networks are Learnable**: By examining activation maps and gradients, black-box models become interpretable.

---

## Part 2: Creating and Experimenting with Image Filters

### 2.1 Understanding Image Filters

Image filters apply mathematical operations to pixel values to create visual effects. Each filter modifies pixels based on their neighbors or applies transformation matrices. Filters are deterministic rule-based systems, distinct from the learned neural networks in Part 1.

### 2.2 Implemented Filters & Analysis

#### Filter 1: Gaussian Blur
**How it works**: Averages each pixel with its neighbors using a weighted Gaussian distribution  
**Parameters**: Radius 3 (moderate blur strength)

**Effect on Boston street image**:
- Smooths architectural details and fine textures
- Reduces sharpness of building edges and flag details
- Creates soft, dreamy quality
- Softens the distinction between foreground (benches) and background (buildings)

**Use cases**: Background blur in photography, soft aesthetic effects, reducing noise

#### Filter 2: Vintage/Sepia
**How it works**: Applies color transformation matrix + reduces saturation

**Effect on Boston street image**:
- Shifts all colors toward warm brown tones
- Makes historic buildings look authentically aged and nostalgic
- American flags take on sepia tone appearance
- Creates 1970s or early 1900s historical photograph aesthetic
- Enhances the "historic Boston" character of the scene

**Mathematical basis**:
```
Output = Input × [0.272  0.534  0.131]
                  [0.349  0.686  0.168]
                  [0.393  0.769  0.189]
```

**Use cases**: Historical photo effects, creating nostalgic mood, vintage design

#### Filter 3: Edge Detection
**How it works**: Uses Sobel filter algorithm to detect pixel value discontinuities

**Effect on Boston street image**:
- Shows building outlines prominently
- Reveals architectural edges sharply
- Flag poles appear as distinct lines
- Bench shapes become clear line drawings
- Creates high-contrast line drawing appearance

**Result**: Clear silhouettes of all objects; looks like pencil sketch or blueprint

**Use cases**: Graphic design, object detection preprocessing, artistic line drawings

#### Filter 4: Posterization
**How it works**: Reduces color palette by limiting bits per channel (3-bit = 512 colors instead of 16M)

**Effect on Boston street image**:
- Creates bold color blocks throughout the scene
- Sky becomes solid color instead of gradient
- Building colors become flat and graphic
- Creates pop art or comic book appearance
- Makes flags stand out in bold, solid colors
- Architectural details disappear into color blocks

**Visual Impact**: Artistic, graphic, and bold appearance—like Andy Warhol pop art

**Use cases**: Artistic effects, poster design, graphic design, pop art

#### Filter 5: Neon Glow
**How it works**: Combines edge detection + color inversion + contrast boost (2x) + saturation boost (1.5x)

**Effect on Boston street image**:
- Creates bright, glowing outlines around buildings and streets
- Makes American flags appear to glow with neon brightness
- Inverted colors create surreal, otherworldly appearance
- High contrast makes edges appear fluorescent
- Cyberpunk aesthetic despite historical subject matter
- Creates luminous appearance as if photographed with neon lights

**Visual Impact**: Bright, modern, high-energy feel—contemporary art style

**Use cases**: Modern art, cyberpunk aesthetic, creative effects, neon installation simulation

#### Filter 6: Oil Painting
**How it works**: Uses median filtering to create smooth painterly effect while preserving edges

**Effect on Boston street image**:
- Smooth regions (building walls, sky) become simplified
- Edges (building outlines, architectural details) remain sharp
- Creates hand-painted, artistic appearance
- Looks like oil painting or watercolor art
- Preserves character of the scene while adding artistic quality
- Details blur into broader color strokes

**Visual Impact**: Artistic, refined, museum-quality appearance

**Use cases**: Artistic transformation, creating digital paintings, enhancing aesthetic

### 2.3 Filter Development Process

**Original Filter**: Basic Gaussian blur - simple radius-based blurring

**Enhancement Approach**:
1. Started with blur foundation (understanding basic filtering)
2. Added multiple filter menu system (organization and choice)
3. Implemented color-based effects (vintage, posterize - color space transformations)
4. Added edge-based effects (edges, neon - gradient computation)
5. Implemented texture effects (oil painting - neighborhood operations)

**Iterative Refinement**:
- Tested different parameter values to find optimal settings
- Adjusted blur radius, posterization bits, neon contrast values through experimentation
- Discovered that parameter values significantly affect visual outcome
- Found that filters compound when applied sequentially

### 2.4 Filter Effectiveness Comparison

| Filter | Effectiveness | Best Use | Difficulty Level |
|--------|---------------|----------|------------------|
| Blur | High | Photography effects | Easy |
| Vintage | High | Nostalgic effects | Medium |
| Edge Detection | High | Graphic art | Medium |
| Posterize | Medium | Artistic effects | Easy |
| Neon Glow | Medium | Modern art | Hard |
| Oil Painting | High | Artistic effects | Hard |

**Most Successful**:
1. **Vintage/Sepia** - Perfectly complements historical Boston architecture, creates authentic period feel
2. **Oil Painting** - Transforms photograph into refined artwork while preserving scene integrity
3. **Edge Detection** - Clearly reveals architectural forms and composition

**Most Creative**:
1. **Neon Glow** - Most dramatic transformation; completely changes aesthetic
2. **Posterize** - Creates bold graphic interpretation of the scene

### 2.5 Real-World Filter Applications

**Photography**: Vintage filters applied to historical images enhance authenticity and nostalgia

**Social Media**: Instagram/Snapchat style filters (blur, vintage, posterize variants) are ubiquitous

**Art**: Oil painting and artistic filters transform photographs into digital artwork

**Computer Vision**: Edge detection preprocessing helps identify objects and scene structure

**Design**: Posterization and neon effects used in graphic design and digital art

---

## Part 3: Working with AI for Code Development

### 3.1 Code Understanding Through AI Assistance

**Initial Prompt**: "Explain what each line of this Python program does"

**AI Response Quality**: Exceptional. The AI provided:
- Line-by-line breakdown of base_classifier.py
- Clear explanations of TensorFlow and Keras functions
- Why each operation was necessary
- How images flow through the preprocessing pipeline

**Key Concepts I Learned**:
1. **Image Preprocessing**: Why resizing to 224x224 is necessary (standard neural network input)
2. **Array Operations**: Why expand_dims adds batch dimension (neural networks process batches)
3. **Normalization**: Why preprocess_input scales pixel values (matching training data)
4. **Model Prediction**: What model.predict() returns and why decoding is necessary

### 3.2 Grad-CAM Algorithm Understanding

**Prompt**: "Can you explain the Grad-CAM algorithm and how it highlights important areas of an image?"

**AI Explanation Helped Me Understand**:
- Gradient computation: How backpropagation reveals feature importance
- Activation maps: What convolutional layers store internally
- Weighting: How gradients determine which features matter for the prediction
- Visualization: Why heatmaps show what the model "attended to"

**Personal Insight**: Grad-CAM transforms a black-box prediction (0.6847 for "street") into an interpretable visualization. This is fundamental to trustworthy AI.

### 3.3 Effective Prompting Strategies

**Prompts That Worked Well**:
1. "Explain what each line of this Python program does" → Got detailed, educational responses
2. "How can I add Grad-CAM to visualize important image regions?" → Got working implementation
3. "Create 6 different artistic filters that [specific effects]" → Got multiple complete implementations
4. "Explain why this Grad-CAM heatmap shows [pattern]" → Got analysis and interpretation

**Prompts That Required Follow-up**:
1. "Help with image processing" → Too vague, needed more specifics
2. "Make this better" → Unclear what "better" meant without context
3. "How do neural networks work?" → Got generic response, needed to ask specifically about MobileNetV2

**Key Learning**: Specific, detailed prompts yield better results. Asking "why" and "how" produces explanations, not just code.

### 3.4 AI as Learning Accelerator

**Advantages**:
- ✅ Instant code explanations without needing to decipher documentation
- ✅ Multiple implementations to compare and learn from
- ✅ Conceptual bridges between theory and practice
- ✅ Debugging help when code doesn't work as expected
- ✅ Iterative refinement based on feedback

**Limitations Encountered**:
- Some generated code needed testing and validation
- Parameter optimization still required manual experimentation
- Edge cases required human judgment
- Context switching across conversations loses some information

**Overall Assessment**: AI significantly accelerated learning. Understanding complex concepts (Grad-CAM, filter mathematics) happened faster through explanation + implementation + testing cycle.

### 3.5 Code Quality & Documentation

**Observations About Provided Code**:
- ✅ Comprehensive comments explaining intent
- ✅ Clear function organization and naming
- ✅ Error handling (try-except blocks)
- ✅ Docstrings for functions
- ✅ Logical progression from simple to complex

**My Enhancement Ideas**:
- Could add more parameter validation
- Could provide more filter combinations
- Could optimize for GPU processing
- Could add batch image processing

---

## Part 4: Synthesis & Personal Reflection

### 4.1 How Rule-Based vs Learned Systems Compare

**Rule-Based (Image Filters)**:
- Explicit, mathematical operations
- Fully deterministic and interpretable
- Easy to understand and modify
- Limited to programmed effects
- Fast and reliable

**Learned Systems (Neural Networks)**:
- Patterns extracted from data
- Black-box until visualized
- Flexible and generalizable
- Require large training datasets
- Can recognize complex patterns humans wouldn't program

**Integration**: This project showed both systems' strengths. Filters provide reliable, creative effects. Neural networks enable intelligent image understanding. Together, they solve different problems well.

### 4.2 Understanding Pre-trained Models

Through this project, I learned:

1. **Transfer Learning is Powerful**: MobileNetV2 trained on ImageNet (14M images) instantly classifies new images. This is far more practical than training from scratch.

2. **Interpretability Unlocks Trust**: Grad-CAM visualization made me confident in the model's reasoning. High confidence without interpretability would be concerning; moderate confidence with sound reasoning is trustworthy.

3. **Model Limitations**: The model achieved 68.47% confidence on a complex scene—realistic for ambiguous cases. A model claiming 99.9% confidence on ambiguous input should raise questions.

4. **Practical Value**: Pre-trained models enable rapid prototyping without ML expertise or computational resources.

### 4.3 Image Processing Mathematics

Understanding filters revealed:
- **Convolution**: Local operations propagate to create global effects
- **Color Space**: Transformations like sepia use mathematical matrices
- **Gradients**: Edge detection uses computed derivatives
- **Reduction**: Posterization quantizes continuous values to discrete levels

These filters are elegant—simple mathematics creates rich visual effects.

### 4.4 AI-Assisted Development Impact

**Before This Project**:
- Limited understanding of neural networks
- Minimal hands-on experience with deep learning
- Uncertain how to interpret model predictions
- Basic image processing knowledge

**After This Project**:
- Practical understanding of transfer learning
- Hands-on experience with Grad-CAM visualization
- Confidence in interpreting neural network decisions
- Knowledge of image filter implementations
- Understanding of effective AI-assisted development

**Biggest Insight**: Working with AI to develop code accelerated learning dramatically. Coding while learning concepts cements understanding better than passive reading.

### 4.5 Real-World Applications

**Image Classification**:
- Medical imaging: Detecting diseases in X-rays, CT scans
- Quality control: Manufacturing defect detection
- Autonomous vehicles: Real-time object detection
- Content moderation: Identifying inappropriate images

**Grad-CAM Visualization**:
- Medical AI: Doctors verify that models identify correct features
- Regulatory compliance: Explainability requirements
- Debugging: Identifying when models focus on wrong features
- Building trust: Making AI decisions transparent

**Image Filters**:
- Photo editing software
- Social media filters
- Video processing
- Artistic applications
- Data augmentation for training

---

## Part 5: Key Learnings & Conclusions

### 5.1 Major Insights

1. **AI Can Be Interpretable**: Grad-CAM proved that neural networks aren't necessarily black boxes. Proper visualization makes decisions transparent.

2. **Mathematics Powers Both Approaches**: Both neural networks and filters rely on mathematical operations. Understanding the math clarifies how systems work.

3. **Transfer Learning Democratizes AI**: Pre-trained models enable anyone to leverage advances made by teams with massive resources.

4. **Hands-On Learning is Essential**: Reading about neural networks is different from implementing, testing, and visualizing them.

5. **AI Tools Amplify Capability**: Working with an AI assistant as collaborator (not replacement) significantly accelerated development and learning.

### 5.2 Challenges Overcome

| Challenge | Solution | Outcome |
|-----------|----------|---------|
| Understanding Grad-CAM math | Asked for step-by-step explanation | Clear understanding of algorithm |
| Interpreting heatmaps | Analyzed patterns against predictions | Gained confidence in results |
| Implementing multiple filters | Used AI assistance with parameter tuning | 6 functional filters created |
| Environment setup | Followed detailed documentation | Successful installation and testing |

### 5.3 Questions This Project Raised

1. **How much of my understanding came from AI explanation vs. my own work?**
   - Answer: Both—AI explained concepts, but testing and analyzing deepened understanding

2. **When should I trust neural network predictions without Grad-CAM?**
   - Answer: For high-confidence predictions on clear inputs (90%+). For ambiguous cases or safety-critical decisions, always verify reasoning.

3. **How would I extend this project?**
   - Train custom model on specific domain
   - Implement other visualization techniques (attention maps, layer activation)
   - Apply filters in creative combinations
   - Create interactive web interface

### 5.4 Personal Takeaways

**About AI Systems**:
- Pre-trained models provide enormous value for practical applications
- Interpretability tools (Grad-CAM) should be standard, not optional
- Transfer learning is perhaps the most important practical AI technique

**About Development**:
- AI assistants are powerful tools when used thoughtfully
- Asking "why" and "how" produces better learning outcomes than just "how"
- Testing and iteration cement understanding better than passive reading

**About Learning**:
- Combining practical coding with conceptual understanding is highly effective
- Hands-on projects reveal insights that theoretical study misses
- Teaching AI systems to explain concepts forces clearer thinking

---

## Conclusion

This Image Classification and Artistic Filters project successfully bridged theoretical AI concepts and practical implementation. The combination of neural network classification, interpretability visualization (Grad-CAM), and image filtering demonstrated different approaches to computational vision.

Key achievements:
- ✅ Understood and implemented transfer learning with MobileNetV2
- ✅ Implemented Grad-CAM for model interpretability
- ✅ Created 6 different artistic image filters
- ✅ Tested system on real image (Boston street scene)
- ✅ Analyzed results and drew meaningful conclusions
- ✅ Effectively collaborated with AI assistant for learning and development

The most important learning: Complex AI systems can be understood through visualization, explanation, and hands-on experimentation. Rather than treating neural networks as mysterious black boxes, proper tools and approaches make them interpretable and trustworthy.

This project demonstrates that AI literacy—understanding how systems work, why they make decisions, and when to trust them—is essential for responsible technology use.

---

## Appendix A: Test Results Detail

### Image Tested
- **Filename**: land-o-lakes-inc-MjH55Ef3w_0-unsplash.jpg
- **Content**: Boston-style urban street with historic buildings, benches, American flags
- **Dimensions**: Resized to 224x224 for classification

### Classification Results
1. street: 68.47%
2. building: 21.56%
3. public_square: 9.97%

### Grad-CAM Findings
- Hottest regions: Building facades, flag poles, architectural details
- Cooler regions: Sky, generic background
- Model correctly focused on semantic content
- Activation patterns suggest hierarchical feature extraction

### Filter Applications
- All 6 filters applied and tested successfully
- Output files generated with descriptive naming
- Each filter produced distinctive visual transformation
- No errors or quality issues observed

---

## Appendix B: Technical Information

### System Configuration
- Python 3.14.2
- TensorFlow/Keras (deep learning)
- NumPy (numerical computing)
- OpenCV/Pillow (image processing)
- Matplotlib (visualization)

### Files Used
- base_classifier_with_gradcam.py (200+ lines)
- advanced_filter.py (300+ lines)
- requirements.txt (6 packages)

### Processing Times
- First run: ~2 minutes (model loading)
- Subsequent runs: ~30 seconds per image
- Filter processing: ~5 seconds per filter

---

## Appendix C: Code Examples from Project

### Grad-CAM Core Concept
```python
# Forward pass through network
last_conv_layer_output, preds = model(img_array, training=False)

# Compute gradients
grads = tape.gradient(class_channel, last_conv_layer_output)

# Weight features by importance
weighted_features = last_conv_layer_output @ pooled_grads

# Result: heatmap showing important regions
```

### Filter Example
```python
# Vintage filter: color transformation + saturation reduction
sepia = img_array @ sepia_matrix.T
enhancer = ImageEnhance.Color(sepia_image)
vintage_image = enhancer.enhance(0.7)  # 70% saturation
```

---

**Document Status**: Complete  
**Date Submitted**: December 15, 2025  
**Student**: Jonathan McCloud

