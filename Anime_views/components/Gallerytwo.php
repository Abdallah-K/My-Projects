



<section id="GalleryTwo">
    <div class="gallerytwobox">
        <div class="gallerycol">

            <div class="gallerycolbox">
                <img class="imgdata" src="views/images/GalleryTwo/mer.jpg">
            </div>
            <div class="gallerycolbox">
                <img class="imgdata"  src="views/images/GalleryTwo/son.jpg">
            </div>
            <div class="gallerycolbox">
                <img class="imgdata"  src="views/images/GalleryTwo/sas.jpg">
            </div>
            <div class="gallerycolbox">
                <img class="imgdata"  src="views/images/GalleryTwo/demon.jpg">
            </div>
        
        </div>
        <div class="gallerydisplay">
            <div id="gallerydisplay-img">
                <img id="disImg" src="views/images/GalleryTwo/mer.jpg">
            </div>
        </div>
            
    </div>

</section>


<script>

    $(function(){

        $(".imgdata").click(function(){
            var img_type = $(this).attr("src") ;
            $("#disImg").attr("src",img_type).hide(0,function(){
                $("#disImg").fadeIn();
            })
        })


    

    
    })

</script>