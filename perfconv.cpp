#include <torch/extension.h>
#include <torch/types.h>
#include <vector>
#include <iostream>

//#include <chrono>

// C++ interface
// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
//#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x " must be a CUDA tensor")
//#define CHECK_CONTIGUOUS(x) AT_CHECK(x.is_contiguous(), #x " must be contiguous")
//#define CHECK_INPUT(x) CHECK_CONTIGUOUS(x)

//using std::chrono::high_resolution_clock;
//using std::chrono::duration;

torch::Tensor kern1d(int64_t k, at::TensorOptions options){
    return k - torch::arange(-k + 1, k, 1, options).abs();
}
torch::Tensor get_lin_kernel(int64_t w, int64_t h, bool normalised, at::TensorOptions options){
    torch::Tensor k = (kern1d(w, options).index({torch::indexing::Slice(0, torch::indexing::None, 1), torch::indexing::None})
                        .matmul(kern1d(h, options)
                        .index({torch::indexing::None,torch::indexing::Slice(0, torch::indexing::None, 1)})))
                    .index({torch::indexing::None,torch::indexing::None,
                            torch::indexing::Slice(),
                            torch::indexing::Slice()}) / (w * h);
    if (normalised){
        return k / k.sum();
    }else{
        return k;
    }
}

std::vector<torch::Tensor> conv_forward(torch::Tensor input,
                                   torch::Tensor weights,
                                   const ::std::optional<at::Tensor> bias,
                                   int64_t kW, int64_t kH,
                                   int64_t dW, int64_t dH, /*stride values*/
                                   int64_t dWp, int64_t dHp, /*perf stride values*/
                                   int64_t padW, int64_t padH, bool is_bias, at::Device device, 
                                   int64_t dilW, int64_t dilH, int64_t groups, bool upscale_conv) {
    //CHECK_INPUT(input);
    //CHECK_INPUT(weights);
    //CHECK_INPUT(bias);
    auto options =
  torch::TensorOptions()
    .dtype(torch::kFloat32)
    .device(input.device());
    int64_t batch_size = input.size(0);
    int64_t nInputPlane = input.size(1);
    int64_t inputWidth = input.size(2);
    int64_t inputHeight = input.size(3);
    //TODO inputWidth pa height sta zamenjana... to je ful confusing za debugiranje, HOWEVER... it works

    int64_t outW = int((inputWidth - ((kW - 1) * dilW) + 2 * padW - 1)  / dW) + 1;
    int64_t outH = int((inputHeight - ((kH - 1) * dilH) + 2 * padH - 1)  / dH) + 1;
    //std::cerr << "before downsampling\n";
    //TODO
    if ((kW == 1 && kH == 1) || (dWp < 2 && dHp < 2)){
    //std::cerr << "using torch impl\n";
        return {at::convolution(input, weights, bias, torch::IntArrayRef({dW, dH}), torch::IntArrayRef({padW, padH}),
                                    torch::IntArrayRef({dilW, dilH}), false /*if transpose conv*/,
                                    torch::IntArrayRef({0, 0}) /*out padding?*/, groups)};
    }
    //kernel size is more than 1 since 1 has no improvement over normal conv
    //dW = dW * dWp;
    //dH = dH * dHp;
    //std::cerr << dW << dH << "\n";
    //std::cerr << input << std::endl;
    //std::cerr << "using manual conv w downscaling\n";
    //auto t1 = high_resolution_clock::now();//TODO we don't need to save to variable, can just use directly?= speedup maybe
    torch::Tensor output = at::convolution(input, weights, bias, torch::IntArrayRef({dW * dWp, dH * dHp}), torch::IntArrayRef({padW, padH}),
                                    torch::IntArrayRef({dilW, dilH}), false /*if transpose conv*/,
                                    torch::IntArrayRef({0, 0}) /*out padding?*/, groups);
    //std::cerr << "after conv\n";
    //TODO test transpose convolution---------------------------------------------------------------------------------
    //std::cerr << dW<<dH<<padW<<padH<<std::endl;
    //std::cerr << "done with reduce"<<std::endl;

    //self.mod1 = ((self.out_x - 1) % self.perf_stride[0]) + 1
    //self.mod2 = ((self.out_y - 1) % self.perf_stride[1]) + 1
    //padding = (self.mod1 - self.n1) % self.mod1, (self.mod2 - self.n2) % self.mod2 // postane 0, ker so n1 0 , torej a % a = 0
    int64_t x = dWp - 1; //+ padding for jitter in interpolation, if we ever do that
    int64_t y = dHp - 1;

    //std::cerr << x << ", " << y << std::endl;
    //std::cerr << output << std::endl;
    //auto a = ;
    //std::cerr << a << std::endl;
    //auto b = a;//TODO tole narobe dela?
    //std::cerr << b << std::endl;
    //auto c = b.view(torch::IntArrayRef({batch_size, output.size(1), outW, outH}));
    if (upscale_conv || (dWp > 2) || (dHp > 2)){
        //std::cerr << "jeba in transpose convolucija\n";
        //std::cerr << outW << outH <<dWp<<dHp <<"\n";
        //std::cerr << dW << "dw, then dh:"<<dH <<"\n";
        //std::cerr << get_lin_kernel(dWp, dHp, false, options) <<"\n";
        //auto a = at::convolution(output.view(torch::IntArrayRef({output.size(0) * output.size(1), 1, output.size(2), output.size(3)})),
        //                    get_lin_kernel(dWp, dHp, false, options), {} /*bias*/,
        //                    torch::IntArrayRef({dWp, dHp}) /*stride*/,
        //                    torch::IntArrayRef({0, 0})/*padding*/, torch::IntArrayRef({dilW, dilH})/*dilation*/, true /*is transpose*/,
        //                    torch::IntArrayRef({0, 0})/*out padding*/, 1);
        //std::cerr << "a: \n" << std::endl;
        //std::cerr << a << std::endl;
        //std::cerr << x << "x and y:"<<y << std::endl;
        //auto b = a.index({torch::indexing::Slice(),
        //                  torch::indexing::Slice(),
        //                        torch::indexing::Slice(x, x+outW),
        //                        torch::indexing::Slice(y, y+outH)});
        //std::cerr << "b: \n" << std::endl;
        //std::cerr << b << std::endl;
        //auto c = b.view(torch::IntArrayRef({batch_size, output.size(1), outW, outH}));
        //std::cerr << "c: \n" << std::endl;
        //std::cerr << c << std::endl;
        //return {c};
        return {at::convolution(output.view(torch::IntArrayRef({output.size(0) * output.size(1), 1, output.size(2), output.size(3)})),
                            get_lin_kernel(dWp, dHp, false, options)/*generated weight*/, {} /*bias*/,
                            torch::IntArrayRef({dWp, dHp}) /*stride*/,
                            torch::IntArrayRef({0, 0})/*padding*/, torch::IntArrayRef({dilW, dilH})/*dilation*/, true /*is transpose*/,
                            torch::IntArrayRef({0, 0})/*out padding*/, 1)
                .index({torch::indexing::Slice(),
                          torch::indexing::Slice(),
                                torch::indexing::Slice(x, x+outW),
                                torch::indexing::Slice(y, y+outH)})
                .view(torch::IntArrayRef({batch_size, output.size(1), outW, outH}))

                            };

    }

    //std::cerr << "after if\n";
    //torch::Tensor out = torch::empty(torch::IntArrayRef({batch_size, nInputPlane, outH, outW}), options);
    torch::Tensor out = torch::zeros(torch::IntArrayRef({batch_size, weights.size(0), outW, outH}), options);
    //std::cerr << "after init out\n";
    //base output
    out.index_put_({torch::indexing::Slice(0, torch::indexing::None, 1),
                torch::indexing::Slice(0, torch::indexing::None, 1),
                torch::indexing::Slice(0, torch::indexing::None, dWp),
                torch::indexing::Slice(0, torch::indexing::None, dHp)}, output);
    //std::cerr << out << std::endl;
    //first right
    //std::cerr << dWp << dHp << std::endl;
    //std::cerr << "have base" << std::endl;

    int64_t lastW = (output.size(2)-1) * dWp;
    int64_t lastH = (output.size(3)-1) * dHp;

    //std::cerr << "before interp" << std::endl;
    if (dHp > 1){
        out.index_put_({
            torch::indexing::Slice(),
            torch::indexing::Slice(),
            torch::indexing::Slice(0, torch::indexing::None, dWp),
            torch::indexing::Slice(1, -(dHp-1), dHp)
            },
            (
                output.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(0, -1, 1)}) +
                output.index({
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(),
                    torch::indexing::Slice(1, torch::indexing::None, 1)})
             )* 0.5);

    }

    //std::cerr << "mid interp" << std::endl;
    out.index_put_(
            {torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(1, -1, 2),
             torch::indexing::Slice()}
        , (out.index({torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(0, -2, 2),
             torch::indexing::Slice()}) +
        out.index({torch::indexing::Slice(),
             torch::indexing::Slice(),
             torch::indexing::Slice(2, torch::indexing::None, 2),
             torch::indexing::Slice()})) *0.5);



    //std::cerr << "before fix_edges" << std::endl;
    //std::cerr << out << std::endl;




    //std::cerr << "after interp vertical, before edge fix" << std::endl;
    //std::cerr << out << std::endl;


    //std::cerr << lastW << lastH << std::endl;
    //std::cerr << outW << outH << std::endl;
    if (lastW != (outW-1)){
        out.index_put_(
                    {
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(lastW+1, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1)
                    }
                , out.index({
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(lastW, lastW+1, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1)}));
    }

    //std::cerr << out << std::endl;
    if (lastH != (outH-1)){
    //std::cerr << "wtf" << std::endl;
        out.index_put_(
                    {
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(lastH+1, torch::indexing::None, 1)
                    }
                , out.index({
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(0, torch::indexing::None, 1),
                    torch::indexing::Slice(lastH, lastH+1, 1)}));
    }
    //t2 = high_resolution_clock::now();
    //ms = t2 - t1;
    //std::cerr << ms.count() << "ms for interpolate\n";
    //std::cerr << out << std::endl;


    return {out};
}



std::vector<torch::Tensor> conv_backward(torch::Tensor input,
                                    torch::Tensor gradOutput,
                                    torch::Tensor weights,
                                    int64_t kW, int64_t kH,
                                    int64_t dW, int64_t dH,/*stride*/
                                    int64_t dWp, int64_t dHp, /*perf stride values*/
                                    int64_t padW, int64_t padH,
                                    bool is_bias, at::Device device, int64_t dilW, int64_t dilH,
                                    int64_t groups, bool stridedBackward) {

    /*CHECK_INPUT(gradOutput);
    CHECK_INPUT(weights);
    CHECK_INPUT(input);
*/

    //int64_t batch_size = input.size(0);
    //int64_t nInputPlane = input.size(1);
    //int64_t inputHeight = input.size(2);
    //int64_t inputWidth = input.size(3);

    int64_t nOutputPlane = gradOutput.size(1);
    //int64_t outputHeight = gradOutput.size(2);
    //int64_t outputWidth = gradOutput.size(3);

    std::array<bool, 3> output_mask = {true, true, true};
    if(stridedBackward){
        //std::cerr << dW << " dW "<< dW * dWp << " dW*dWp "<< dW/dWp << " dW/dWp \n";
        //std::cerr << dH << " dH "<< dH * dHp << " dH*dHp "<< dH/dHp << " dH/dHp \n";
        int64_t strideb1 = dW * dWp;
        int64_t strideb2 = dH * dHp;
        auto backTest = at::convolution_backward(gradOutput.index({torch::indexing::Slice(),torch::indexing::Slice(),
                                                                          torch::indexing::Slice(0, torch::indexing::None, dWp),
                                                                          torch::indexing::Slice(0, torch::indexing::None, dHp)}),
                                        input,
                                        weights,
                                torch::IntArrayRef({nOutputPlane}) /*nOutputPlane*/,
                                torch::IntArrayRef({strideb1, strideb2})/*stride*/,
                                torch::IntArrayRef({padW, padH})/*padding*/,
                                torch::IntArrayRef({dilW, dilH})/*dilation*/, false /*if transpose conv*/,
                                torch::IntArrayRef({0, 0}) /*out padding?*/, groups, output_mask);
        if (is_bias){
            return {std::get<0>(backTest),std::get<1>(backTest),std::get<2>(backTest)};
        }else{
            return {std::get<0>(backTest),std::get<1>(backTest),{}};
        }
    }else{
        //std::cerr << gradOutput << std::endl;
        //std::cerr << input << std::endl;
        //std::cerr << out << std::endl;
        auto backTest = at::convolution_backward(gradOutput,
                                        input,
                                        weights,
                                torch::IntArrayRef({nOutputPlane}) /*nOutputPlane*/,
                                torch::IntArrayRef({dW, dH})/*stride*/,
                                torch::IntArrayRef({padW, padH}),
                                torch::IntArrayRef({dilW, dilH}), false /*if transpose conv*/,
                                torch::IntArrayRef({0, 0}) /*out padding?*/, groups, output_mask);
        if (is_bias){
            return {std::get<0>(backTest),std::get<1>(backTest),std::get<2>(backTest)};
        }else{
            return {std::get<0>(backTest),std::get<1>(backTest),{}};
        }
    }

    /*std::cerr << stride << std::endl;
    std::cerr << gradOutput << "\n" << gradOutput.sizes() << std::endl;
    std::cerr << input << "\n" << input.sizes() << std::endl;
    std::cerr << weights << "\n" << weights.sizes() << std::endl;
    std::cerr << stride << std::endl;
    std::cerr << nOutputPlane << std::endl;
    std::cerr << padW << padH << std::endl;*/





}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &conv_forward, "conv forward");
  m.def("backward", &conv_backward, "conv backward");
}