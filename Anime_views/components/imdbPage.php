


<section id="IMDBPage">
    <div class="imdbox">
        <div class="imdbtitle">
            <h1>Top IMDB Anime Rate</h1>
        </div>
        <div class="imdblist">

        </div>

    </div>
</section>

<script>

    $(function(){
        

        var out = "";
        $.post("API/PostModel.php",{
            action:"select",
            tablename:"imdbdata"
        },function(data){
            data = $.parseJSON(data);
            $.each(data,(index,outitem)=>{
                out +=`
                    <div class="imdbcard">
                        <img src="views/images/IMDB/${outitem.image}">
                        <div class="overflow">
                            <div class="over-one">
                                <div class="rate">
                                    <h2>${outitem.rate}</h2>
                                </div>
                            </div>
                            <div class="over-two">
                                <a href="./imdbCard.php?id=${outitem.id}">See More</a>
                            </div>
                        </div>
                    </div>`;
            })
            $(".imdblist").html(out);
        })


    })

</script>
