

<section id="GalleryThree">
    <div class="galbox">
        <div class="galdis">
            <div class="galdiscard">
                <img id="imageDis" src="views/images/GalleryThree/jiren.png">
            </div>
        </div>
        <div class="galslide">
            
            <div class="galbtn">
                <button id="galPrev"><i class="fa-solid fa-arrow-left"></i></button>
            </div>
            <div class="galslider">


                <div class="galslidecard">
                    <img  class="img_src" src="views/images/GalleryThree/jiren.png">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/kaido.png">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/hash.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/shanks.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/kid.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/death.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/hero.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/levi.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/one.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/black.jpg">
                </div>
                <div class="galslidecard">
                    <img class="img_src" src="views/images/GalleryThree/boruto.jpg">
                </div>







            </div>
            <div class="galbtn">
                <button id="galNext"><i class="fa-solid fa-arrow-right"></i></button>
            </div>

        </div>

    </div>

</section>



<script>

    $(function(){

        $("#galNext").click(()=>{
            var pos = $(".galslider").scrollLeft() + 200;
            $(".galslider").animate({scrollLeft:pos},200);
        })

        $("#galPrev").click(()=>{
            var pos = $(".galslider").scrollLeft() - 200;
            $(".galslider").animate({scrollLeft:pos},200);
        })


        $(".img_src").click(function(){
            var img_type = $(this).attr("src");
            $("#imageDis").attr("src",img_type).hide(0,function(){
                $("#imageDis").fadeIn(500);
            })
        })




        
    })

</script>