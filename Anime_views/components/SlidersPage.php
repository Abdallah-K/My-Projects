




<section id="Sliders">
    <div class="latestAnime">
        <div class="latestBox">
            <div class="latestTitle">
                <h1>Famous Anime</h1>
            </div>
            <div class="latestsec">
                <div class="latestbtn">
                    <button id="Prevbtn"><i class="fa-solid fa-arrow-left"></i></button>
                </div>
                <div class="latestSlider">


                </div>
                <div class="latestbtn">
                    <button id="Nextbtn"><i class="fa-solid fa-arrow-right"></i></button>
                </div>
            </div>

        </div>

    </div>
    <div class="banners">
        <div class="bannerBox">
            <div class="bannertitle">
                <h1>Banners</h1>
            </div>
            <div class="bannerinfo">
                <div class="bannerbtn">
                    <button id="bannerPrev"><i class="fa-solid fa-arrow-left"></i></button>
                </div>
                <div class="baanerslider">

                    <!-- <div class="bannerCard">
                        <img src="views/images/banners/naruto.jpg">
                    </div> -->
                    


                </div>
                <div class="bannerbtn">
                    <button id="bannerNext"><i class="fa-solid fa-arrow-right"></i></button>
                </div>


            </div>
        </div>
    </div>
</section>



<script>

    $(function(){
        $("#Nextbtn").click(()=>{
            let pos = $(".latestSlider").scrollLeft() + 200;
            $('.latestSlider').animate({scrollLeft:pos},200);
        })

        $("#Prevbtn").click(()=>{
            let pos = $(".latestSlider").scrollLeft() - 200;
            $('.latestSlider').animate({scrollLeft:pos},200);
        })
        
        var posNb = 500;
        $("#bannerPrev").click(()=>{
            let pos = $(".baanerslider").scrollLeft() - posNb;
            $('.baanerslider').animate({scrollLeft:pos},posNb);
        })

        $("#bannerNext").click(()=>{
            let pos = $(".baanerslider").scrollLeft() + posNb;
            $('.baanerslider').animate({scrollLeft:pos},posNb);
        })



        var bannerout = "";
        $.post("API/PostModel.php",{
            action:"select",
            tablename:"bannersdata",
        },function(data){
            data = $.parseJSON(data);
            $.each(data,(index,banneritem)=>{
                bannerout +=`
                    <div class="bannerCard">
                        <img src="views/images/banners/${banneritem.image}">
                    </div>
                `;
            })
            $(".baanerslider").html(bannerout);
        })



        var famousCard = "";
        $.post("API/PostModel.php",{
            action:"select",
            tablename:"famousdata",
        },function(data){
            data = $.parseJSON(data);
            $.each(data,(index,famousitem)=>{
                famousCard +=`
                    <div class="latestCard">
                        <img src="views/images/anime/${famousitem.image}">
                    </div>`;
            })
            $(".latestSlider").html(famousCard);
        })



    })

</script>