#include "renderer.h"
#include "model.h"

// basic file operations
#include <iostream>
#include <fstream>

//npy
#include <cnpy.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow *window, Renderer &renderer, int* imgCounter, 
                  const std::string rgb_dir, const std::string pose_dir,
                  const std::string rgb_suffix = "png", const std::string pose_suffix = "txt");
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
bool takeScreenshot = false;
bool spacePressedAtLeastOnce = false;

// camera
Camera camera(glm::vec3(0.790932f, 1.300000f, 1.462270f)); // 1.3705f, 1.51739f, 1.44963f    0.0f, 0.0f, 3.0f      -0.3f, 0.3f, 0.3f    0.790932f, 1.300000f, 1.462270f
float lastX = DEF_WIDTH / 2.0f;
float lastY = DEF_HEIGHT / 2.0f;
bool firstMouse = true;

// timing
float deltaTime = 0.0f;
float lastFrame = 0.0f;

Renderer::Renderer(string const &pathToMesh, int width, int height, int shader_mode, const std::string rgb_texture) : shader_mode(shader_mode), rgb_texture(rgb_texture) {
    m_buffer_width = width;
    m_buffer_height = height;

    if(init()){
        // if init fails, then the return code is != 0 which is equal to this if statement
        throw std::runtime_error("Failed to init renderer");
    }

    m_model = new Model(pathToMesh);
    
    if (shader_mode == uvmap) {
        m_shader = new Shader("../shader/uvmap.vs", "../shader/uvmap.frag");
    } else if (shader_mode == angle) {
        m_shader = new Shader("../shader/angle.vs", "../shader/angle.frag");
    } else if (shader_mode == depth) {
        m_shader = new Shader("../shader/depth.vs", "../shader/depth.frag");
    } else if (shader_mode == rgb) {
        m_shader = new Shader("../shader/rgb.vs", "../shader/rgb.frag");
    } else if (shader_mode == vertex_color || shader_mode == pose || shader_mode == copy_color || shader_mode == copy_depth) {
        m_shader = new Shader("../shader/vertex_color.vs", "../shader/vertex_color.frag");
    } else {
        throw std::runtime_error("Unsupported shader_mode: " + shader_mode);
    }
}

Renderer::~Renderer() {
    //delete &m_model;
    //delete &m_shader;
    glfwTerminate();
}

int Renderer::init() {
    if(! glfwInit()){
        std::cout << "Failed to init glfw" << std::endl;
        return EXIT_FAILURE;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    
#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // uncomment this statement to fix compilation on OS X
#endif

    // glfw window creation
    // --------------------
    m_window = glfwCreateWindow(m_buffer_width, m_buffer_height, "Matterport_Renderer", NULL, NULL);
    if (m_window == nullptr) {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return EXIT_FAILURE;
    }
    glfwMakeContextCurrent(m_window);

    // To avoid: https://stackoverflow.com/questions/8302625/segmentation-fault-at-glgenvertexarrays-1-vao
    glewExperimental = GL_TRUE; 
    if (GLEW_OK != glewInit()){
        std::cout << "Failed to init glew" << std::endl;
        return EXIT_FAILURE;
    }

    glfwSetFramebufferSizeCallback(m_window, framebuffer_size_callback);
    glfwSetCursorPosCallback(m_window, mouse_callback);
    glfwSetScrollCallback(m_window, scroll_callback);
    glfwSetKeyCallback(m_window, key_callback);

    // configure global opengl state
    glEnable(GL_DEPTH_TEST);

     // Define a texture with mipmap
    glGenTextures(1,&origin_color);
    glActiveTexture(GL_TEXTURE0); // activate the texture unit first before binding texture
	glBindTexture(GL_TEXTURE_2D, origin_color);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_LOD, 3.0);
    // glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_LOD_BIAS, 1.0);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // load the texture
    int width=1024, height=1024, nrChannels=3;
    if(rgb_texture != ""){
        unsigned char *data = stbi_load(rgb_texture.c_str(), &width, &height, &nrChannels, 0);
        if(! data){
            throw std::runtime_error("Failed to load texture from: " + rgb_texture);
        }

        GLfloat value, max_anisotropy = 8.0f; /* don't exceed this value...*/
        glGetFloatv(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT, & value);

        value = (value > max_anisotropy) ? max_anisotropy : value;
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_MAX_ANISOTROPY_EXT, value);

        glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0,GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
        stbi_image_free(data);
    } else {
        // glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, m_buffer_width*2, m_buffer_height*2, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);
	    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB, width, height, 0,GL_RGB, GL_UNSIGNED_BYTE, 0);
        glGenerateMipmap(GL_TEXTURE_2D);
    }

    // Define Framebuffer and add a float texture to it that will store the rendered uvs
	glGenFramebuffers(1,&fbo);
	glBindFramebuffer(GL_FRAMEBUFFER,fbo);
	
	glGenTextures(1,&image_uv);
    glActiveTexture(GL_TEXTURE1); // activate the texture unit first before binding texture
	glBindTexture(GL_TEXTURE_2D, image_uv);
	glTexImage2D(GL_TEXTURE_2D, 0,GL_RGB32F, m_buffer_width, m_buffer_height, 0,GL_RGB, GL_FLOAT, 0);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D,image_uv, 0);

    // need to add a renderbuffer with depth, otherwise opengl will not perform depth test in the framebuffer
    glGenRenderbuffers(1, &rbo);
    glBindRenderbuffer(GL_RENDERBUFFER, rbo);
    glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, m_buffer_width, m_buffer_height);
    glBindRenderbuffer(GL_RENDERBUFFER, 0);
    glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, rbo);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);   

    m_initialized = true;
    return 0;
}

void Renderer::render(const glm::mat4& model, const glm::mat4& view, const glm::mat4& projection){
    if(! m_initialized){
        std::cout << "Cannot render before initializing the renderer" << std::endl;
        return;
    }
    // Draw to output window --> uvs as 8-Bit RGB values
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    m_shader->use();
    m_shader->setMat4("projection", projection);
    m_shader->setMat4("view", view);
    m_shader->setMat4("model", model);
    m_shader->setInt("texture_rgb", 0);
    m_model->draw(*m_shader);

    if(shader_mode != rgb && shader_mode != vertex_color){
        // Draw to output float texture via the Framebuffer --> uvs as 32-Bit float values
        glBindFramebuffer(GL_FRAMEBUFFER, fbo);

        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        m_shader->use();
        m_shader->setMat4("projection", projection);
        m_shader->setMat4("view", view);
        m_shader->setMat4("model", model);
        m_shader->setInt("texture_rgb", 0);
        m_model->draw(*m_shader);
    }
}

void Renderer::savePose(std::string filename, const glm::mat4& view){
    ofstream cam_file;
    cam_file.open(filename);
    char line[50];
    sprintf(line, "%.6f %.6f %.6f %.6f\n", view[0][0], view[1][0], view[2][0], view[3][0]);
    cam_file << line;
    sprintf(line, "%.6f %.6f %.6f %.6f\n", view[0][1], view[1][1], view[2][1], view[3][1]);
    cam_file << line;
    sprintf(line, "%.6f %.6f %.6f %.6f\n", view[0][2], view[1][2], view[2][2], view[3][2]);
    cam_file << line;
    sprintf(line, "%.6f %.6f %.6f %.6f\n", view[0][3], view[1][3], view[2][3], view[3][3]);
    cam_file << line;
    cam_file.close();
}

void Renderer::saveIntrinsics(std::string filename, const glm::mat3& intr, int width, int height){
    ofstream cam_file;
    cam_file.open(filename);
    char line[50];
    sprintf(line, "%.6f %.6f %.6f\n", intr[0][0], intr[0][1], intr[0][2]);
    cam_file << line;
    sprintf(line, "%.6f %.6f %.6f\n", intr[1][0], intr[1][1], intr[1][2]);
    cam_file << line;
    sprintf(line, "%.6f %.6f %.6f\n", intr[2][0], intr[2][1], intr[2][2]);
    cam_file << line;
    sprintf(line, "%d %d\n", width, height);
    cam_file << line;
    cam_file.close();
}

void Renderer::saveUV(std::string filename, bool flip) {
    GLfloat * image_uv_data = new GLfloat[m_buffer_height * m_buffer_width * 3];

    glReadBuffer(GL_COLOR_ATTACHMENT0);
	glReadPixels(0, 0, m_buffer_width, m_buffer_height, GL_RGB, GL_FLOAT, image_uv_data);
	
    // keep the R and G channel, as they capture the uv coordinates. B is zero everywhere and can be discarded
    // write this in a sequential vector that will then be saved as a npy array
	std::vector<float> data;
	for (int j = 0; j < m_buffer_height; j++){
		for (int i = 0; i < m_buffer_width; i++){
			int t = j * m_buffer_width * 3 + i * 3;
            if(flip){
                t = (m_buffer_height - 1 - j) * m_buffer_width * 3 + i * 3;
            }
			data.push_back(image_uv_data[t]);
			data.push_back(image_uv_data[t+1]);
            data.push_back(image_uv_data[t+2]);
		}
	}
	
    // cnpy::npy_save(filename,&data[0],{(unsigned long)m_buffer_height,(unsigned long)m_buffer_width,2},"w");
    cnpy::npy_save(filename,&data[0],{(unsigned long)m_buffer_height,(unsigned long)m_buffer_width,3},"w");

    // cnpy::npz_save(zipname, filename, &data[0], {(unsigned long) m_buffer_height, (unsigned long) m_buffer_width, 2}, "a");

	delete[] image_uv_data;
}

void Renderer::readRGB(cv::Mat& image, bool flip) {
    // glBindFramebuffer(GL_FRAMEBUFFER, 1);
    // glBindFramebuffer(GL_FRAMEBUFFER, fbo);

    glReadBuffer(GL_COLOR_ATTACHMENT0);

    image = cv::Mat(m_buffer_height, m_buffer_width, CV_8UC3); //32FC3
    
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_PACK_ALIGNMENT, (image.step & 3) ? 1 : 4);

    //set length of one complete row in destination data (doesn't need to equal img.cols)
    glPixelStorei(GL_PACK_ROW_LENGTH, image.step/image.elemSize());

    glReadPixels(0, 0, image.cols, image.rows, GL_BGR, GL_UNSIGNED_BYTE, image.data);

    //glReadPixels(0, 0, image.cols, image.rows, GL_BGR, GL_FLOAT, image.data);

    if(flip){
        cv::flip(image, image, 0);
    }
}

void Renderer::readDepth(cv::Mat& image, bool flip) {
    image = cv::Mat(m_buffer_height, m_buffer_width, CV_16UC1);
    std::vector<float> data_buff(m_buffer_height * m_buffer_width);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadPixels(0, 0, m_buffer_width, m_buffer_height, GL_DEPTH_COMPONENT, GL_FLOAT, data_buff.data());
    for (int i = 0; i < m_buffer_height; i++) {
        for (int j = 0; j < m_buffer_width; j++) {
            int t = i * m_buffer_width + j;
            if(flip){
                t = (m_buffer_height - 1 - i) * m_buffer_width + j;
            }
            const float zn = (2 * data_buff[static_cast<int>(t)] - 1);
            const float ze = (2 * kFarPlane * kNearPlane) / (kFarPlane + kNearPlane + zn*(kNearPlane - kFarPlane));
            image.at<unsigned short>(m_buffer_height - i - 1, j) = ze;
        }
    }
}

void Renderer::renderInteractive(glm::mat4 start_extrinsics, glm::mat4 projection,
                                const std::string rgb_dir, const std::string pose_dir,
                                const std::string rgb_suffix, const std::string pose_suffix) {
    // tell GLFW to capture our mouse
    glfwSetInputMode(m_window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
    
    // render loop
    int imgCounter = 0;
    bool firstImage = true;

    while (!glfwWindowShouldClose(m_window))
    {

        // per-frame time logic
        // --------------------
        float currentFrame = glfwGetTime();
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // input
        // -----
        processInput(m_window, *this, &imgCounter, rgb_dir, pose_dir, rgb_suffix, pose_suffix);
        
        // model/view/projection transformations
        // ------
        // glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)m_buffer_width / (float)m_buffer_height, 0.1f, 100.0f);
        glm::mat4 view = camera.GetViewMatrix();
        
        if(firstImage){
            camera.Position = glm::vec3(start_extrinsics[3][0], start_extrinsics[3][1], start_extrinsics[3][2]);
            camera.Position = glm::vec3(0.0f, 0.0f, 0.0f);
            firstImage = false;
            view = camera.GetViewMatrix();
        }
        

        // render
        // ------
        render(view, glm::mat4(1.0f), projection);

        // glfw: swap buffers and poll IO events (keys pressed/released, mouse moved etc.)
        // -------------------------------------------------------------------------------
        glfwSwapBuffers(m_window);
        glfwPollEvents();
    }
}

// process all input: query GLFW whether relevant keys are pressed/released this frame and react accordingly
// ---------------------------------------------------------------------------------------------------------
void processInput(GLFWwindow *window, Renderer &renderer, int* imgCounter,
                  const std::string rgb_dir, const std::string pose_dir,
                  const std::string rgb_suffix, const std::string pose_suffix) {
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);

    if (takeScreenshot){
        cv::Mat colorImage;
        renderer.readRGB(colorImage);

        // save matrix as file
        if (!colorImage.empty()) {
            std::stringstream image_filename;
            char image_name[30];
            sprintf(image_name, "%d.%s", *imgCounter, rgb_suffix.c_str());
            image_filename << rgb_dir << "/" << image_name;
            cv::imwrite(image_filename.str(), colorImage);

            std::cout << "Wrote image: " << image_name << std::endl;

            // write cam matrix
            std::stringstream cam_filename;
            char cam_name[30];
            sprintf(cam_name, "%d.%s", *imgCounter, pose_suffix.c_str());
            cam_filename << pose_dir << "/" << cam_name;

            glm::mat4 view = camera.GetViewMatrix();
            // view = glm::inverse(view); // RT goes from world to view, but in ICL we save view-to-world so use this camera here as well.

            ofstream cam_file;
            cam_file.open (cam_filename.str());
            char line[50];
            sprintf(line, "%.6f %.6f %.6f %.6f\n", camera.Right[0], camera.Up[0], camera.Front[0], camera.Position[0]);
            cam_file << line;
            sprintf(line, "%.6f %.6f %.6f %.6f\n", camera.Right[1], camera.Up[1], camera.Front[1], camera.Position[1]);
            cam_file << line;
            sprintf(line, "%.6f %.6f %.6f %.6f\n", camera.Right[2], camera.Up[2], camera.Front[2], camera.Position[2]);
            cam_file << line;
            sprintf(line, "%.6f %.6f %.6f %.6f\n", 0.0f, 0.0f, 0.0f, 1.0f);
            cam_file << line;
            cam_file.close();

            std::cout << "Wrote cam: " << cam_name << std::endl;

            // increment
            (*imgCounter)++;
        }

        takeScreenshot = false;
    }
}

void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if ( (key == GLFW_KEY_SPACE && action == GLFW_PRESS) 
       ||(key == GLFW_KEY_W && spacePressedAtLeastOnce && action == GLFW_PRESS)
       ||(key == GLFW_KEY_A && spacePressedAtLeastOnce && action == GLFW_PRESS)
       ||(key == GLFW_KEY_S && spacePressedAtLeastOnce && action == GLFW_PRESS)
       ||(key == GLFW_KEY_D && spacePressedAtLeastOnce && action == GLFW_PRESS)){
        takeScreenshot = true;
    }

    if (! spacePressedAtLeastOnce && key == GLFW_KEY_SPACE && action == GLFW_PRESS){
        spacePressedAtLeastOnce = true;
    }
}

// glfw: whenever the window size changed (by OS or user resize) this callback function executes
// ---------------------------------------------------------------------------------------------
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    // make sure the viewport matches the new window dimensions; note that width and 
    // height will be significantly larger than specified on retina displays.
    glViewport(0, 0, width, height);
}

// glfw: whenever the mouse moves, this callback is called
// -------------------------------------------------------
void mouse_callback(GLFWwindow* window, double xpos, double ypos)
{
    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos; // reversed since y-coordinates go from bottom to top

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

// glfw: whenever the mouse scroll wheel scrolls, this callback is called
// ----------------------------------------------------------------------
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(yoffset);
}